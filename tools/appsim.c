/* appsim.c - reproduce the iOS app's exact prompt/KV-cache flow on the desktop.
 *
 *   ./appsim model.sllm tokenizer.tok "你好" [more turns...]
 *
 * The app keeps one token buffer and one KV cache across turns: it appends the
 * user turn, prefills only the new tokens, then appends the sampled ids plus
 * <|im_end|>\n for the next turn.  Anything that goes wrong there (an off-by-one
 * in the prefill point, a stale position) shows up here and not in the CLI.
 */
#include "sllm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int *conv;
static int  convN, convCap;
static SllmTok tok;

static void push_id(int id) { if (convN < convCap) conv[convN++] = id; }
static void push_text(const char *s) { if (convN < convCap) convN += tok_encode(&tok, s, conv + convN, convCap - convN); }

static void system_prefix(void)
{
    convN = 0;
    push_id(tok.special[1]);
    push_text("system\n");
    push_text("You are a helpful assistant");
    push_id(tok.special[2]);
    push_text("\n");
}

static void out_cb(void *ud, const char *bytes, int len)
{
    int *last = (int *)ud;
    (void)last;
    fwrite(bytes, 1, (size_t)len, stdout);
    fflush(stdout);
}

static int last_tok;
static void stderr_log(const char *line) { fprintf(stderr, "| %s\n", line); }
static void tok_cb(void *ud, const char *bytes, int len)
{
    (void)ud; (void)bytes; (void)len;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "usage: %s model.sllm tok.bin question [question...]\n", argv[0]);
        return 1;
    }
    if (tok_load(&tok, argv[2]) != 0) return 1;

    SllmModel m;
    if (sllm_load(&m, argv[1], 4) != 0) return 1;
    if (getenv("SLLM_LOG")) sllm_set_log(stderr_log);

    SllmSampler smp;
    sampler_init(&smp, m.cfg.vocab_size);
    smp.temp = 0.85f; smp.top_k = 40; smp.top_p = 0.90f; smp.rep_pen = 1.05f;

    convCap = m.cfg.max_seq;
    conv = (int *)calloc((size_t)convCap, sizeof(int));
    int *gen = (int *)calloc((size_t)convCap, sizeof(int));
    system_prefix();
    int convKv = 0;
    sllm_reset(&m);

    for (int turn = 3; turn < argc; turn++) {
        const char *q = argv[turn];
        int need = (int)strlen(q) + 24 + 96 + 8;
        if (convN + need >= convCap) { system_prefix(); convKv = 0; sllm_reset(&m); }
        int prefillFrom = convKv;

        push_id(tok.special[1]);
        push_text("user\n");
        push_text(q);
        push_id(tok.special[2]);
        push_text("\n");
        push_id(tok.special[1]);
        push_text("assistant\n");

        int promptLen = convN;
        int maxNew = 96;
        if (maxNew > convCap - promptLen - 4) maxNew = convCap - promptLen - 4;
        if (maxNew < 1) maxNew = 1;

        printf("[turn %d] prompt=%d cache=%d max_new=%d\n", turn - 2, promptLen, prefillFrom, maxNew);
        printf("Q: %s\nA: ", q);
        int n = sllm_generate(&m, &tok, &smp, conv, promptLen, prefillFrom,
                              maxNew, out_cb, NULL, gen, maxNew);
        printf("\n");
        last_tok = n > 0 ? gen[n - 1] : -1;
        printf("[ids]");
        for (int i = 0; i < n; i++) printf(" %d", gen[i]);
        printf("\n[last] id=%d  model.pos=%d  convN=%d\n\n", last_tok, m.pos, promptLen);

        convN = promptLen;
        for (int i = 0; i < n; i++) push_id(gen[i]);
        push_id(tok.special[2]);
        push_text("\n");
        convKv = m.pos;
        if (convKv >= convCap - 8) { system_prefix(); convKv = 0; sllm_reset(&m); }
        (void)tok_cb;
    }
    return 0;
}
