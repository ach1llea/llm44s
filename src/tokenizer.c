/* tokenizer.c - GPT-2 style byte-level BPE, id-keyed merge table */
#include "sllm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rd_u32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static int32_t  rd_i32(const uint8_t *p) { return (int32_t)rd_u32(p); }

int tok_load(SllmTok *t, const char *path)
{
    memset(t, 0, sizeof(*t));
    FILE *f = fopen(path, "rb");
    if (!f) { perror("open tok"); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); return -1; }
    fclose(f);

    if (memcmp(buf, "STOK", 4) != 0) { free(buf); fprintf(stderr, "bad tok magic\n"); return -1; }
    const uint8_t *p = buf + 8;
    t->n_vocab   = rd_i32(p); p += 4;
    t->n_merges  = rd_i32(p); p += 4;
    int n_spec   = rd_i32(p); p += 4;
    p += 4;
    for (int i = 0; i < 256; i++) { t->byte_id[i] = rd_i32(p); p += 4; }
    t->mkeys = (uint32_t *)malloc(sizeof(uint32_t) * t->n_merges);
    t->mvals = (uint32_t *)malloc(sizeof(uint32_t) * t->n_merges);
    for (int i = 0; i < t->n_merges; i++) { t->mkeys[i] = rd_u32(p); p += 4; }
    for (int i = 0; i < t->n_merges; i++) { t->mvals[i] = rd_u32(p); p += 4; }
    t->special[0] = t->special[1] = t->special[2] = -1;
    for (int i = 0; i < n_spec && i < 3; i++) { t->special[i] = rd_i32(p); p += 4; }

    t->tok    = (uint8_t **)calloc(t->n_vocab, sizeof(uint8_t *));
    t->toklen = (uint16_t *)calloc(t->n_vocab, sizeof(uint16_t));
    const uint8_t *end = buf + sz;
    for (int i = 0; i < t->n_vocab; i++) {
        if (p + 2 > end) { fprintf(stderr, "truncated vocab at %d\n", i); return -1; }
        int len = p[0] | (p[1] << 8);
        p += 2;
        t->tok[i] = (uint8_t *)p;
        t->toklen[i] = (uint16_t)len;
        p += len;
    }

    t->hsize = 1;
    while (t->hsize < t->n_merges * 2) t->hsize <<= 1;
    t->hash = (int32_t *)calloc(t->hsize, sizeof(int32_t));
    for (int i = 0; i < t->n_merges; i++) {
        uint32_t key = t->mkeys[i] + 1;              /* +1 so 0 means empty */
        uint32_t h = (key * 2654435761u) & (t->hsize - 1);
        while (t->hash[h]) h = (h + 1) & (t->hsize - 1);
        t->hash[h] = (int32_t)i + 1;                 /* store index+1 */
    }
    t->mbase = buf;
    return 0;
}

void tok_free(SllmTok *t)
{
    free(t->tok); free(t->toklen); free(t->mkeys); free(t->mvals);
    free(t->hash); free(t->mbase);
    memset(t, 0, sizeof(*t));
}

/* returns the merge *rank* (index into the merge table) or -1 */
static int merge_rank(SllmTok *t, int a, int b)
{
    uint32_t key = ((uint32_t)a << 13) | (uint32_t)b;
    uint32_t h = ((key + 1) * 2654435761u) & (t->hsize - 1);
    while (t->hash[h]) {
        int idx = t->hash[h] - 1;
        if (t->mkeys[idx] == key) return idx;
        h = (h + 1) & (t->hsize - 1);
    }
    return -1;
}

int tok_piece(SllmTok *t, int id, const uint8_t **bytes)
{
    if (id < 0 || id >= t->n_vocab) { *bytes = NULL; return 0; }
    *bytes = t->tok[id];
    return t->toklen[id];
}

/* --- GPT-2-ish pre-tokenisation ----------------------------------------- */
enum { C_SPACE, C_LETTER, C_DIGIT, C_OTHER };

static int utf8_decode(const uint8_t *s, int n, int *cp)
{
    uint8_t c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && n >= 2) { *cp = ((c & 0x1F) << 6) | (s[1] & 0x3F); return 2; }
    if ((c & 0xF0) == 0xE0 && n >= 3) { *cp = ((c & 0x0F) << 12) | ((s[1] & 0x3F) << 6) | (s[2] & 0x3F); return 3; }
    if ((c & 0xF8) == 0xF0 && n >= 4) { *cp = ((c & 0x07) << 18) | ((s[1] & 0x3F) << 12) | ((s[2] & 0x3F) << 6) | (s[3] & 0x3F); return 4; }
    *cp = c; return 1;
}

static int classify(int cp)
{
    if (cp == ' ') return C_SPACE;
    if (cp == 0x09 || cp == 0x0A || cp == 0x0D || cp == 0x0B || cp == 0x0C) return C_SPACE;
    if (cp < 0x80) {
        if ((cp >= 'a' && cp <= 'z') || (cp >= 'A' && cp <= 'Z')) return C_LETTER;
        if (cp >= '0' && cp <= '9') return C_DIGIT;
        return C_OTHER;
    }
    /* CJK punctuation / fullwidth ASCII punctuation behave like \p{P} */
    if (cp >= 0x3000 && cp <= 0x303F) return C_OTHER;
    if ((cp >= 0xFF01 && cp <= 0xFF0F) || (cp >= 0xFF1A && cp <= 0xFF20) ||
        (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65)) return C_OTHER;
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201C || cp == 0x201D ||
        cp == 0x2026 || cp == 0x2014 || cp == 0x2013) return C_OTHER;
    return C_LETTER;
}

static int bpe_chunk(SllmTok *t, const uint8_t *s, int n, int *out, int max_out)
{
    if (n <= 0) return 0;
    int *sym = (int *)malloc(sizeof(int) * (size_t)n);
    int ns = 0;
    if (!sym) return 0;
    for (int i = 0; i < n; i++) sym[ns++] = t->byte_id[s[i]];
    while (ns > 1) {
        int best = -1, bestrank = 1 << 30;
        for (int i = 0; i + 1 < ns; i++) {
            int r = merge_rank(t, sym[i], sym[i + 1]);
            if (r < 0) continue;
            if (r < bestrank) { bestrank = r; best = i; }
        }
        if (best < 0) break;
        sym[best] = (int)t->mvals[bestrank];
        for (int i = best + 1; i + 1 < ns; i++) sym[i] = sym[i + 1];
        ns--;
    }
    int k = ns < max_out ? ns : max_out;
    for (int i = 0; i < k; i++) out[i] = sym[i];
    free(sym);
    return k;
}

int tok_encode(SllmTok *t, const char *text, int *ids, int max_ids)
{
    const uint8_t *s = (const uint8_t *)text;
    int n = (int)strlen(text), i = 0, nout = 0;

    while (i < n && nout < max_ids) {
        int cp;
        utf8_decode(s + i, n - i, &cp);
        int cls = classify(cp);

        if (cls == C_SPACE) {
            int j = i;
            while (j < n) {
                int c2, l2 = utf8_decode(s + j, n - j, &c2);
                if (classify(c2) != C_SPACE) break;
                j += l2;
            }
            /* GPT-2: `\s+(?!\S)` eats all but the last space, then ` ?\p{L}+`
               takes that last space plus the following run. */
            int attach = (j < n) && (s[j - 1] == ' ');
            if (attach) {
                int k = i + (j - i > 1 ? j - i - 1 : 0);
                if (k > i) nout += bpe_chunk(t, s + i, k - i, ids + nout, max_ids - nout);
                int c2;
                utf8_decode(s + j, n - j, &c2);
                int cls2 = classify(c2);
                int e = j;
                while (e < n) {
                    int c3, l3 = utf8_decode(s + e, n - e, &c3);
                    if (classify(c3) != cls2) break;
                    e += l3;
                }
                nout += bpe_chunk(t, s + k, e - k, ids + nout, max_ids - nout);
                i = e;
            } else {
                nout += bpe_chunk(t, s + i, j - i, ids + nout, max_ids - nout);
                i = j;
            }
        } else {
            int k = i;
            while (k < n) {
                int c2, l2 = utf8_decode(s + k, n - k, &c2);
                if (classify(c2) != cls) break;
                k += l2;
            }
            nout += bpe_chunk(t, s + i, k - i, ids + nout, max_ids - nout);
            i = k;
        }
    }
    return nout;
}
