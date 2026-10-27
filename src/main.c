/* main.c - CLI driver */
#include "sllm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static long rss_kb(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}

static void out_cb(void *ud, const char *bytes, int len)
{
    (void)ud;
    fwrite(bytes, 1, (size_t)len, stdout);
    fflush(stdout);
}

static int build_chat(SllmTok *t, const char *sys, const char *user, int *ids, int cap)
{
    int n = 0;
#define PUSH_ID(x) do { if (n < cap) ids[n++] = (x); } while (0)
#define PUSH_TX(x) do { if (n < cap) n += tok_encode(t, (x), ids + n, cap - n); } while (0)
    PUSH_ID(t->special[1]);
    PUSH_TX("system\n");
    PUSH_TX(sys);
    PUSH_ID(t->special[2]);
    PUSH_TX("\n");
    PUSH_ID(t->special[1]);
    PUSH_TX("user\n");
    PUSH_TX(user);
    PUSH_ID(t->special[2]);
    PUSH_TX("\n");
    PUSH_ID(t->special[1]);
    PUSH_TX("assistant\n");
#undef PUSH_ID
#undef PUSH_TX
    return n;
}

static void stderr_log(const char *line) { fprintf(stderr, "%s\n", line); }

static void usage(const char *p)
{
    fprintf(stderr,
        "usage: %s -m model.sllm -t tok.bin [options]\n"
        "  -p TEXT        prompt (single turn)\n"
        "  -n N           max new tokens (default 96)\n"
        "  --system TEXT  system prompt\n"
        "  --temp F       temperature, 0 = greedy (default 0.85)\n"
        "  --top-k N      top-k (default 50)\n"
        "  --top-p F      top-p (default 0.90)\n"
        "  --rep F        repetition penalty (default 1.05)\n"
        "  --threads N    worker threads (default 1)\n"
        "  -v             print the engine log (the in-app console lines)\n"
        "  --serve PORT   run the HTTP+SSE server instead\n"
        "  --web DIR      directory holding index.html (default .)\n"
        "  --tok-test F   tokenize each line of file F, print ids\n"
        "  --bench        run a short throughput benchmark\n", p);
}

int main(int argc, char **argv)
{
    const char *mpath = NULL, *tpath = NULL, *prompt = NULL;
    const char *system = "You are a helpful assistant", *web = ".";
    const char *toktest = NULL, *logits_out = NULL;
    int max_new = 96, threads = 1, serve = 0, bench = 0, port = 8080, verbose = 0;
    float temp = 0.85f, top_p = 0.90f, rep = 1.05f;
    int top_k = 50;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-m") && i + 1 < argc) mpath = argv[++i];
        else if (!strcmp(a, "-t") && i + 1 < argc) tpath = argv[++i];
        else if (!strcmp(a, "-p") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(a, "-n") && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (!strcmp(a, "--system") && i + 1 < argc) system = argv[++i];
        else if (!strcmp(a, "--temp") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(a, "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(a, "--top-p") && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (!strcmp(a, "--rep") && i + 1 < argc) rep = (float)atof(argv[++i]);
        else if (!strcmp(a, "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--serve") && i + 1 < argc) { serve = 1; port = atoi(argv[++i]); }
        else if (!strcmp(a, "--web") && i + 1 < argc) web = argv[++i];
        else if (!strcmp(a, "--tok-test") && i + 1 < argc) toktest = argv[++i];
        else if (!strcmp(a, "--logits-out") && i + 1 < argc) logits_out = argv[++i];
        else if (!strcmp(a, "--bench")) bench = 1;
        else if (!strcmp(a, "-v")) verbose = 1;
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 0; }
    }
    if (!mpath || !tpath) { usage(argv[0]); return 1; }

    if (verbose) sllm_set_log(stderr_log);

    SllmTok tok;
    if (tok_load(&tok, tpath) != 0) return 1;

    if (toktest) {
        FILE *f = fopen(toktest, "r");
        if (!f) { perror("test file"); return 1; }
        char line[4096];
        int *ids = (int *)calloc(4096, sizeof(int));
        while (fgets(line, sizeof(line), f)) {
            size_t L = strlen(line);
            while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
            if (!L) continue;
            int n = tok_encode(&tok, line, ids, 4096);
            printf("%s\t", line);
            for (int i = 0; i < n; i++) printf("%d%s", ids[i], i + 1 < n ? "," : "");
            printf("\n");
        }
        fclose(f);
        return 0;
    }

    SllmModel m;
    if (sllm_load(&m, mpath, threads) != 0) return 1;
    fprintf(stderr, "[model] dim=%d layers=%d heads=%d kv=%d ffn=%d vocab=%d ctx=%d\n",
            m.cfg.dim, m.cfg.n_layers, m.cfg.n_heads, m.cfg.n_kv_heads,
            m.cfg.hidden_dim, m.cfg.vocab_size, m.cfg.max_seq);
    fprintf(stderr, "[model] threads=%d rss=%ldKB\n", threads, rss_kb());

    SllmSampler smp;
    sampler_init(&smp, m.cfg.vocab_size);
    smp.temp = temp; smp.top_k = top_k; smp.top_p = top_p; smp.rep_pen = rep;

    if (serve)
        return http_serve(&m, &tok, port, web, system);

    int cap = m.cfg.max_seq;
    int *ids = (int *)calloc((size_t)cap, sizeof(int));
    int *gen = (int *)calloc((size_t)cap, sizeof(int));

    const char *text = prompt ? prompt : (bench ? "你好" : NULL);
    if (!text) { usage(argv[0]); return 1; }

    if (max_new > cap - 48) max_new = cap - 48;
    int np = build_chat(&tok, system, text, ids, cap);
    fprintf(stderr, "[prompt] %d tokens\n", np);

    /* prefill */
    sllm_reset(&m);
    double t0 = now_ms();
    const float *logits = NULL;
    for (int i = 0; i < np; i++) logits = sllm_forward(&m, ids[i]);
    double t_pre = now_ms() - t0;

    if (logits_out && logits) {
        FILE *lf = fopen(logits_out, "wb");
        if (lf) {
            fwrite(logits, sizeof(float), (size_t)m.cfg.vocab_size, lf);
            fclose(lf);
            fprintf(stderr, "[dump] logits -> %s (%d floats)\n", logits_out, m.cfg.vocab_size);
        }
    }

    /* decode */
    int produced = 0;
    int *recent = (int *)calloc((size_t)cap, sizeof(int));
    int nr = 0;
    for (int i = 0; i < np; i++) recent[nr++] = ids[i];

    double t1 = now_ms();
    for (int i = 0; i < max_new; i++) {
        if (m.pos >= cap - 1) break;
        int tk = sampler_pick(&smp, (float *)logits, recent, nr);
        if (tk == tok.special[2] || tk == tok.special[0]) break;
        const uint8_t *bytes;
        int len = tok_piece(&tok, tk, &bytes);
        out_cb(NULL, (const char *)bytes, len);
        gen[produced++] = tk;
        if (nr < cap) recent[nr++] = tk;
        logits = sllm_forward(&m, tk);
    }
    double t_dec = now_ms() - t1;
    printf("\n");

    double tps = produced > 0 ? produced * 1000.0 / t_dec : 0.0;
    fprintf(stderr, "\n[stats] prefill %d tok in %.0f ms (%.1f tok/s)\n",
            np, t_pre, np * 1000.0 / (t_pre > 0.1 ? t_pre : 0.1));
    fprintf(stderr, "[stats] decode  %d tok in %.0f ms (%.2f tok/s, %.1f ms/tok)\n",
            produced, t_dec, tps, produced > 0 ? t_dec / produced : 0.0);
    fprintf(stderr, "[stats] flops   %.1f MFLOP/token\n",
            (double)m.flops / (double)(np + produced > 0 ? np + produced : 1) / 1e6);
    fprintf(stderr, "[stats] rss     %ld KB\n", rss_kb());
    return 0;
}
