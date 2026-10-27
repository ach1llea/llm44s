/* sampler.c - temperature / top-k / top-p / repetition penalty, bounded memory */
#include "sllm.h"

#include <math.h>
#include <string.h>

#define MAXK 128

void sampler_init(SllmSampler *s, int vocab)
{
    s->vocab = vocab;
    s->temp = 0.85f;
    s->top_k = 50;
    s->top_p = 0.85f;
    s->rep_pen = 1.05f;
    s->rng = 0x9E3779B97F4A7C15ULL;
}

static float frand(SllmSampler *s)
{
    uint64_t x = s->rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    s->rng = x;
    return (float)((x >> 40) & 0xFFFFFF) * (1.0f / 16777216.0f);
}

int sampler_pick(SllmSampler *s, float *logits, const int *recent, int n_recent)
{
    int V = s->vocab;
    int greedy = s->temp <= 0.0f;      /* temp 0 means "reproducible", i.e. argmax */
    float temp = greedy ? 1.0f : s->temp;

    /* Penalise each distinct token once, over a fixed window.  The previous
       version divided once per *occurrence* in the whole context, so a token
       that had already appeared a dozen times (的, 我, a newline) was suppressed
       by 1.05^12 = 1.8x and the model was pushed onto increasingly rare words.
       That is not the "repetition penalty" anyone means to ship. */
    if (s->rep_pen != 1.0f && n_recent > 0) {
        int w = n_recent < 64 ? n_recent : 64;
        for (int i = n_recent - w; i < n_recent; i++) {
            int id = recent[i];
            if (id < 0 || id >= V) continue;
            int seen = 0;
            for (int j = i + 1; j < n_recent && !seen; j++)
                if (recent[j] == id) seen = 1;
            if (seen) continue;
            float v = logits[id];
            logits[id] = v > 0 ? v / s->rep_pen : v * s->rep_pen;
        }
    }

    int K = s->top_k > 0 ? s->top_k : 0;
    if (greedy) {
        int best = 0;
        float bv = -1e30f;
        for (int i = 0; i < V; i++)
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
    if (K > MAXK) K = MAXK;
    if (K == 0 || K > V) K = V < MAXK ? V : MAXK;

    float kv[MAXK];
    int   ki[MAXK];
    int   kc = 0;
    float kmin = -1e30f;

    for (int i = 0; i < V; i++) {
        float v = logits[i] / temp;            /* NaN never compares > kmin */
        if (kc < K) {
            int pos = kc++;
            while (pos > 0 && kv[pos - 1] < v) { kv[pos] = kv[pos - 1]; ki[pos] = ki[pos - 1]; pos--; }
            kv[pos] = v; ki[pos] = i;
            kmin = (kc < K) ? -1e30f : kv[K - 1];
        } else if (v > kmin) {
            int pos = K - 1;
            while (pos > 0 && kv[pos - 1] < v) { kv[pos] = kv[pos - 1]; ki[pos] = ki[pos - 1]; pos--; }
            kv[pos] = v; ki[pos] = i;
            kmin = kv[K - 1];
        }
    }
    if (kc == 0) return 0;

    float mx = kv[0];
    float sum = 0.0f;
    for (int i = 0; i < kc; i++) { kv[i] = expf(kv[i] - mx); sum += kv[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < kc; i++) kv[i] *= inv;

    int keep = kc;
    if (s->top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < kc; i++) {
            cum += kv[i];
            if (cum >= s->top_p) { keep = i + 1; break; }
        }
        if (keep < 1) keep = 1;
    }

    float r = frand(s);
    float acc = 0.0f;
    for (int i = 0; i < keep; i++) {
        acc += kv[i];
        if (r <= acc) return ki[i];
    }
    return ki[keep - 1];
}
