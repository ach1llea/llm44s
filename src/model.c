/* model.c - .sllm loader + Llama-architecture forward pass (batch 1, KV cache) */
#include "sllm.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define SLLM_ALIGN(x, a) (((x) + ((a) - 1)) & ~((size_t)(a) - 1))

static void *xalloc(size_t n)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, SLLM_ALIGN(n, 64)) != 0) return NULL;
    memset(p, 0, n);
    return p;
}

int sllm_probe(const char *path, SllmProbe *out)
{
    memset(out, 0, sizeof(*out));
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    uint8_t h[68];
    ssize_t got = read(fd, h, sizeof(h));
    struct stat st;
    int ok = (got == (ssize_t)sizeof(h)) && memcmp(h, "SLLM", 4) == 0 &&
             fstat(fd, &st) == 0;
    if (!ok) { close(fd); return -1; }

    /* scan the tensor directory so the picker can say "int4/int8" for a file
       that mixes both; costs one small read on a 68-byte-header probe */
    uint32_t nt = 0;
    memcpy(&nt, h + 48, 4);
    if (nt > 0 && nt < 4096) {
        uint8_t *dir = (uint8_t *)malloc((size_t)nt * 36);
        if (dir) {
            int n8 = 0, n4 = 0;
            if (pread(fd, dir, (size_t)nt * 36, 68) == (ssize_t)nt * 36) {
                for (uint32_t i = 0; i < nt; i++) {
                    int32_t ndim, r, c;
                    uint32_t nb;
                    memcpy(&ndim, dir + i * 36 + 4, 4);
                    memcpy(&r,    dir + i * 36 + 8, 4);
                    memcpy(&c,    dir + i * 36 + 12, 4);
                    memcpy(&nb,   dir + i * 36 + 24, 4);
                    if (ndim < 2 || r <= 0 || c <= 0) continue;
                    if (nb == (uint32_t)r * (uint32_t)c) n8++; else n4++;
                }
            }
            free(dir);
            out->mixed = (n8 > 0 && n4 > 0);
        }
    }
    close(fd);

    memcpy(&out->dim,        h + 8,  4);
    memcpy(&out->n_layers,   h + 12, 4);
    memcpy(&out->n_heads,    h + 16, 4);
    memcpy(&out->n_kv_heads, h + 20, 4);
    memcpy(&out->head_dim,   h + 24, 4);
    memcpy(&out->hidden_dim, h + 28, 4);
    memcpy(&out->vocab_size, h + 32, 4);
    memcpy(&out->max_seq,    h + 36, 4);
    memcpy(&out->quant,      h + 52, 4);
    memcpy(&out->qgroup,     h + 56, 4);
    out->file_size = (long)st.st_size;

    /* parameter count: embeddings + per-layer (q,k,v,o,gate,up,down) */
    int hd = out->head_dim ? out->head_dim : out->dim / (out->n_heads ? out->n_heads : 1);
    long kv = (long)out->n_kv_heads * hd;
    long per = (long)out->dim * out->dim * 2            /* q, o  */
             + 2 * (long)kv * out->dim                   /* k, v  */
             + 3 * (long)out->hidden_dim * out->dim;     /* mlp   */
    out->n_params = (long)out->vocab_size * out->dim + per * out->n_layers;
    return 0;
}

/* Log lines are read on a 320 pt wide screen with a monospaced font, so every
   format string here is deliberately kept under ~30 characters: one event, one
   line, no wrapping. */
static const char *base_name(const char *path)
{
    const char *s = strrchr(path, '/');
    return s ? s + 1 : path;
}

int sllm_load(SllmModel *m, const char *path, int n_threads)
{
    memset(m, 0, sizeof(*m));
    sllm_log("open  %s", base_name(path));
    int fd = open(path, O_RDONLY);
    if (fd < 0) { sllm_log("FAIL  cannot open model"); return -1; }
    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }
    void *base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) { perror("mmap"); return -1; }

    uint8_t *p = (uint8_t *)base;
    if (memcmp(p, "SLLM", 4) != 0) { fprintf(stderr, "bad magic\n"); return -1; }
    uint32_t ver = *(uint32_t *)(p + 4);
    if (ver != 1) { fprintf(stderr, "bad version %u\n", ver); return -1; }

    SllmConfig *c = &m->cfg;
    memcpy(&c->dim,         p + 8,  4);
    memcpy(&c->n_layers,    p + 12, 4);
    memcpy(&c->n_heads,     p + 16, 4);
    memcpy(&c->n_kv_heads,  p + 20, 4);
    memcpy(&c->head_dim,    p + 24, 4);
    memcpy(&c->hidden_dim,  p + 28, 4);
    memcpy(&c->vocab_size,  p + 32, 4);
    memcpy(&c->max_seq,     p + 36, 4);
    memcpy(&c->rms_eps,     p + 40, 4);
    memcpy(&c->rope_theta,  p + 44, 4);
    uint32_t n_tensors = *(uint32_t *)(p + 48);
    c->quant = (int)*(uint32_t *)(p + 52);
    c->qgroup = (int)*(uint32_t *)(p + 56);      /* 0 for pre-v1.1 files */
    if (c->quant == Q_I4 && c->qgroup <= 0) c->qgroup = 64;
    sllm_log("size  %.0f MB, %u tensors", (double)st.st_size / 1048576.0, n_tensors);
    sllm_log("quant %s", c->quant == Q_I4 ? "int4 (+int8)" :
                         c->quant == Q_I8 ? "int8" : "fp32");
    sllm_log("dim   %d  L%d  H%d/%d", c->dim, c->n_layers, c->n_heads, c->n_kv_heads);
    sllm_log("cfg   ffn%d v%d c%d", c->hidden_dim, c->vocab_size, c->max_seq);

    m->base = p;
    m->size = (size_t)st.st_size;
    m->n_tensors = (int)n_tensors;
    m->tensors = (SllmTensor *)(p + 68);
    int map_n = 1 + c->n_layers * L_N_PER_LAYER + 1;
    m->tmap = (int32_t *)xalloc(sizeof(int32_t) * map_n);
    for (int i = 0; i < map_n; i++) m->tmap[i] = -1;
    for (uint32_t i = 0; i < n_tensors; i++) {
        int id = m->tensors[i].id;
        if (id >= 0 && id < map_n) m->tmap[id] = (int32_t)i;
    }
    if (m->tmap[0] < 0) { fprintf(stderr, "model has no token embedding\n"); return -1; }

    int dim = c->dim, hd = c->head_dim, kvst = c->n_kv_heads * hd;
    int big = dim > c->hidden_dim ? dim : c->hidden_dim;

    m->x      = (float *)xalloc(sizeof(float) * dim);
    m->xb     = (float *)xalloc(sizeof(float) * dim);
    m->att    = (float *)xalloc(sizeof(float) * dim);
    m->tmp    = (float *)xalloc(sizeof(float) * big);
    m->gate   = (float *)xalloc(sizeof(float) * c->hidden_dim);
    m->up     = (float *)xalloc(sizeof(float) * c->hidden_dim);
    m->kvb    = (float *)xalloc(sizeof(float) * kvst);
    m->scores = (float *)xalloc(sizeof(float) * c->max_seq);
    m->xq     = (int8_t *)xalloc(sizeof(int8_t) * big);
    m->logits = (float *)xalloc(sizeof(float) * c->vocab_size);
    m->cos_tab = (float *)xalloc(sizeof(float) * c->max_seq * (hd / 2));
    m->sin_tab = (float *)xalloc(sizeof(float) * c->max_seq * (hd / 2));
    size_t kvbytes = (size_t)c->n_layers * c->max_seq * kvst;
    m->kcache = (float *)xalloc(sizeof(float) * kvbytes);
    m->vcache = (float *)xalloc(sizeof(float) * kvbytes);
    if (!m->x || !m->logits || !m->kcache || !m->cos_tab) {
        fprintf(stderr, "out of memory allocating state\n");
        return -1;
    }

    int half = hd / 2;
    for (int pos = 0; pos < c->max_seq; pos++) {
        for (int i = 0; i < half; i++) {
            double inv = pow((double)c->rope_theta, -(double)(2 * i) / (double)hd);
            double a = (double)pos * inv;
            m->cos_tab[pos * half + i] = (float)cos(a);
            m->sin_tab[pos * half + i] = (float)sin(a);
        }
    }

    m->pos = 0;
    m->flops = 0;
    m->n_threads = n_threads;
    m->pool = pool_create(n_threads);
    m->n_threads = n_threads;          /* pool may have capped it */
    sllm_log("state kv%.1fMB th%d", (double)(2 * kvbytes * sizeof(float)) / 1048576.0,
             m->n_threads);
    sllm_log("ready rss%ldMB free%ldMB", sllm_rss_kb() / 1024, sllm_free_mb());
    return 0;
}

void sllm_free(SllmModel *m)
{
    pool_free(m->pool);
    if (m->base) munmap(m->base, m->size);
    free(m->tmap); free(m->x); free(m->xb); free(m->att); free(m->tmp);
    free(m->gate); free(m->up); free(m->kvb); free(m->scores); free(m->xq); free(m->logits);
    free(m->cos_tab); free(m->sin_tab); free(m->kcache); free(m->vcache);
    m->base = NULL;
}

void sllm_reset(SllmModel *m) { m->pos = 0; }

const float *sllm_tensor_f32(SllmModel *m, int tid, int *n)
{
    int slot = m->tmap[tid];
    if (slot < 0) return NULL;
    SllmTensor *t = &m->tensors[slot];
    if (n) *n = t->shape[0] * (t->ndim > 1 ? t->shape[1] : 1);
    return (const float *)(m->base + t->off);
}

void mat_get(SllmModel *m, int tid, SllmMat *mat)
{
    SllmTensor *t = &m->tensors[m->tmap[tid]];
    mat->rows = t->shape[0];
    mat->cols = t->shape[1];
    mat->w    = m->base + t->off;
    mat->group = m->cfg.qgroup;
    /* Per-tensor precision, recovered from the byte count: one byte per weight
       is int8, half a byte is int4.  This lets a single file mix both (attention
       and the tied head at 8 bits, the parameter-heavy MLP at 4) and keeps the
       old homogeneous files readable without any format change. */
    mat->quant = (t->nbytes == (uint32_t)(mat->rows * mat->cols)) ? Q_I8 : Q_I4;
    if (mat->quant == Q_I4) {
        mat->sb = (const uint16_t *)(m->base + t->soff);
        mat->sf = NULL;
    } else {
        mat->sf = (const float *)(m->base + t->soff);
        mat->sb = NULL;
    }
}

static void qmv_t(SllmModel *m, int tid, const int8_t *x, float xscale, float *out)
{
    SllmMat mat;
    mat_get(m, tid, &mat);
    qmv_mat(m, &mat, x, xscale, out);
}

static void rmsnorm(float *o, const float *x, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float scale = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * scale * w[i];
}

static void rope_apply(float *v, const float *cosv, const float *sinv, int nh, int hd)
{
    int half = hd / 2;
    for (int h = 0; h < nh; h++) {
        float *q = v + h * hd;
        for (int i = 0; i < half; i++) {
            float a = q[i], b = q[i + half];
            float c = cosv[i], s = sinv[i];
            q[i]      = a * c - b * s;
            q[i + half] = b * c + a * s;
        }
    }
}

static float dotf(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static inline float silu(float v) { return v / (1.0f + expf(-v)); }

static void softmax_inplace(float *p, int n)
{
    float mx = p[0];
    for (int i = 1; i < n; i++) if (p[i] > mx) mx = p[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { p[i] = expf(p[i] - mx); sum += p[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) p[i] *= inv;
}

const float *sllm_forward(SllmModel *m, int token)
{
    SllmConfig *c = &m->cfg;
    int dim = c->dim, hd = c->head_dim, nh = c->n_heads, nkv = c->n_kv_heads;
    int nrep = nh / nkv, half = hd / 2;
    int kvst = nkv * hd;
    int pos = m->pos;
    if (pos >= c->max_seq) pos = c->max_seq - 1;   /* stay in range; caller caps */

    {
        SllmMat emb;
        mat_get(m, T_TOK_EMB, &emb);
        mat_row_f32(&emb, token, m->x);
    }

    const float *cosv = m->cos_tab + (size_t)pos * half;
    const float *sinv = m->sin_tab + (size_t)pos * half;
    float att_scale = 1.0f / sqrtf((float)hd);

    for (int l = 0; l < c->n_layers; l++) {
        int b = 1 + l * L_N_PER_LAYER;

        rmsnorm(m->xb, m->x, sllm_tensor_f32(m, b + L_ATTN_NORM, NULL), dim, c->rms_eps);
        float xs = quantize_act(m->xq, m->xb, dim);

        float *q = m->tmp;                 /* dim floats, reused below */
        qmv_t(m, b + L_WQ, m->xq, xs, q);
        qmv_t(m, b + L_WK, m->xq, xs, m->att);      /* kvst floats */
        qmv_t(m, b + L_WV, m->xq, xs, m->kvb);      /* kvst floats */

        float *kb = m->kcache + (size_t)l * c->max_seq * kvst + (size_t)pos * kvst;
        float *vb = m->vcache + (size_t)l * c->max_seq * kvst + (size_t)pos * kvst;
        for (int i = 0; i < kvst; i++) vb[i] = m->kvb[i];

        rope_apply(q, cosv, sinv, nh, hd);
        rope_apply(m->att, cosv, sinv, nkv, hd);
        for (int i = 0; i < kvst; i++) kb[i] = m->att[i];

        for (int h = 0; h < nh; h++) {
            int kvh = h / nrep;
            const float *qh = q + h * hd;
            int nt = pos + 1;
            for (int t = 0; t < nt; t++) {
                const float *kt = m->kcache + (size_t)l * c->max_seq * kvst
                                + (size_t)t * kvst + kvh * hd;
                m->scores[t] = dotf(qh, kt, hd) * att_scale;
            }
            softmax_inplace(m->scores, nt);
            float *oh = m->att + h * hd;
            for (int d = 0; d < hd; d++) oh[d] = 0.0f;
            for (int t = 0; t < nt; t++) {
                float p = m->scores[t];
                const float *vt = m->vcache + (size_t)l * c->max_seq * kvst
                                + (size_t)t * kvst + kvh * hd;
                for (int d = 0; d < hd; d++) oh[d] += p * vt[d];
            }
        }

        float xs2 = quantize_act(m->xq, m->att, dim);
        qmv_t(m, b + L_WO, m->xq, xs2, m->xb);
        for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];

        rmsnorm(m->xb, m->x, sllm_tensor_f32(m, b + L_FFN_NORM, NULL), dim, c->rms_eps);
        float xs3 = quantize_act(m->xq, m->xb, dim);
        qmv_t(m, b + L_W1, m->xq, xs3, m->gate);
        qmv_t(m, b + L_W3, m->xq, xs3, m->up);
        for (int i = 0; i < c->hidden_dim; i++) m->tmp[i] = silu(m->gate[i]) * m->up[i];
        float xs4 = quantize_act(m->xq, m->tmp, c->hidden_dim);
        qmv_t(m, b + L_W2, m->xq, xs4, m->xb);
        for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];
    }

    rmsnorm(m->xb, m->x, sllm_tensor_f32(m, 1 + c->n_layers * L_N_PER_LAYER, NULL),
            dim, c->rms_eps);
    float xsf = quantize_act(m->xq, m->xb, dim);
    qmv_t(m, T_TOK_EMB, m->xq, xsf, m->logits);

    m->pos = pos + 1;
    return m->logits;
}

int sllm_generate(SllmModel *m, SllmTok *t, SllmSampler *s,
                  const int *prompt, int n_prompt, int prefill_from,
                  int max_new, token_cb cb, void *ud, int *out_ids, int out_cap)
{
    int *recent = (int *)xalloc(sizeof(int) * m->cfg.max_seq);
    if (!recent) return -1;
    int n_recent = 0, produced = 0;

    const float *logits = NULL;
    if (n_prompt <= 0) return -1;
    if (prefill_from < 0) prefill_from = 0;

    double t_start = sllm_now_ms();
    sllm_log("gen   p%d c%d n%d", n_prompt, prefill_from, max_new);
    for (int i = 0; i < n_prompt; i++)
        if (n_recent < m->cfg.max_seq) recent[n_recent++] = prompt[i];
    for (int i = prefill_from; i < n_prompt; i++) {
        if (m->pos >= m->cfg.max_seq - 1) break;
        logits = sllm_forward(m, prompt[i]);
    }
    if (!logits) return -1;
    {
        int np = n_prompt - prefill_from;
        double dt = sllm_now_ms() - t_start;
        sllm_log("pre   %dt %.0fms %.0f/s", np, dt, dt > 0.5 ? np * 1000.0 / dt : 0.0);
    }

    int run = 0;                       /* consecutive identical tokens */
    const char *why = "max";
    for (int i = 0; i < max_new; i++) {
        if (m->pos >= m->cfg.max_seq - 1) { why = "ctx"; break; }
        int tok = sampler_pick(s, (float *)logits, recent, n_recent);
        if (tok == t->special[2] || tok == t->special[0]) { why = "eos"; break; }
        /* A 100M model left to its own devices occasionally locks onto one
           token and repeats it forever; eight in a row is never real text. */
        if (out_ids && produced && tok == out_ids[produced - 1]) {
            if (++run >= 7) { why = "loop"; break; }
        } else {
            run = 0;
        }
        const uint8_t *bytes;
        int len = tok_piece(t, tok, &bytes);
        if (len > 0 && cb) cb(ud, (const char *)bytes, len);
        if (out_ids && produced < out_cap) out_ids[produced] = tok;
        if (n_recent < m->cfg.max_seq) recent[n_recent++] = tok;
        produced++;
        double t_tok = sllm_now_ms();
        logits = sllm_forward(m, tok);
        sllm_log("tok %3d %-6d %4.1fms c%d", produced, tok,
                 sllm_now_ms() - t_tok, m->pos);
    }
    {
        double dt = sllm_now_ms() - t_start;
        sllm_log("done  %dt %.1fs %.0f/s %s", produced, dt / 1000.0,
                 dt > 0.5 ? produced * 1000.0 / dt : 0.0, why);
    }
    free(recent);
    return produced;
}
