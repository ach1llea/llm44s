/* quant.c - int8 x int16 dot products (NEON + scalar) and a tiny thread pool */
#include "sllm.h"

#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define SLLM_NEON 1
#endif

/* ---------------------------------------------------------------- kernels */
/* Activations are quantized to the same +-127 range as the weights, so both
   sides fit in int8.  vmull_s8 gives 8 exact 16-bit products, and vpadalq_s16
   folds them into a 32-bit accumulator.  8 MACs per instruction pair, and the
   pairwise 16-bit add cannot overflow (127*127*2 < 32767). */
#if SLLM_NEON
static inline int32_t hsum_s32(int32x4_t v)
{
#if defined(__aarch64__)
    return vaddvq_s32(v);
#else
    int32x2_t t = vadd_s32(vget_low_s32(v), vget_high_s32(v));
    t = vpadd_s32(t, t);
    return vget_lane_s32(t, 0);
#endif
}
#endif

int32_t dot_i8_i8(const int8_t *w, const int8_t *x, int n)
{
#if SLLM_NEON
    /* Two accumulators, not one: vpadalq has ~4 cycle latency, so a single
       chain stalls the pipeline for every 8 MACs.  Alternating hides it. */
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        int8x16_t w0 = vld1q_s8(w + i);
        int8x16_t w1 = vld1q_s8(w + i + 16);
        int8x16_t x0 = vld1q_s8(x + i);
        int8x16_t x1 = vld1q_s8(x + i + 16);
        a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(w0),  vget_low_s8(x0)));
        a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
        a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(w1),  vget_low_s8(x1)));
        a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
    }
    int32_t s = hsum_s32(vaddq_s32(a0, a1));
    for (; i < n; i++) s += (int32_t)w[i] * (int32_t)x[i];
    return s;
#else
    int32_t s = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s += (int32_t)w[i] * x[i] + (int32_t)w[i + 1] * x[i + 1]
           + (int32_t)w[i + 2] * x[i + 2] + (int32_t)w[i + 3] * x[i + 3];
    }
    for (; i < n; i++) s += (int32_t)w[i] * x[i];
    return s;
#endif
}

/* ------------------------------------------------------------ int4 kernel */
/* Weight layout per 32-value sub-block: 16 bytes, byte[i] holds element i in
   its low nibble and element 16+i in its high nibble.  Nibbles are signed
   two's complement (-8..7), so sign extension is two shifts. */
int32_t dot_i4_i8(const uint8_t *qs, const int8_t *x, int nsub)
{
#if SLLM_NEON
    int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
    for (int s = 0; s < nsub; s++) {
        uint8x16_t p = vld1q_u8(qs + s * 16);
        /* shl 4 puts the nibble's sign bit in bit7, the arithmetic shift back
           sign-extends it */
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(p), 4), 4);
        int8x16_t hi = vshrq_n_s8(vreinterpretq_s8_u8(p), 4);
        const int8_t *xx = x + s * 32;
        int8x16_t xa = vld1q_s8(xx);
        int8x16_t xb = vld1q_s8(xx + 16);
        a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(lo),  vget_low_s8(xa)));
        a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(lo), vget_high_s8(xa)));
        a0 = vpadalq_s16(a0, vmull_s8(vget_low_s8(hi),  vget_low_s8(xb)));
        a1 = vpadalq_s16(a1, vmull_s8(vget_high_s8(hi), vget_high_s8(xb)));
    }
    return hsum_s32(vaddq_s32(a0, a1));
#else
    int32_t total = 0;
    for (int s = 0; s < nsub; s++) {
        const uint8_t *q = qs + s * 16;
        const int8_t *xx = x + s * 32;
        for (int i = 0; i < 32; i++) {
            int v = (i < 16) ? (q[i] & 0x0F) : (q[i - 16] >> 4);
            if (v >= 8) v -= 16;
            total += v * (int32_t)xx[i];
        }
    }
    return total;
#endif
}

float quantize_act(int8_t *dst, const float *src, int n)
{
    float amax = 0.0f;
    int i;
    for (i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float scale = amax / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f / scale;
    for (i = 0; i < n; i++) {
        float v = src[i] * inv;
        int32_t q = (int32_t)(v + (v >= 0 ? 0.5f : -0.5f));
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        dst[i] = (int8_t)q;
    }
    return scale;
}

/* ------------------------------------------------------------ thread pool */
struct SllmPool {
    int n_threads;
    pthread_t th[SLLM_MAX_THREADS];
    void     *wargs[SLLM_MAX_THREADS];
    pthread_mutex_t mu;
    pthread_cond_t  cv_go, cv_done;
    int generation, arrive, shutdown;
    void (*fn)(void *arg, int tid, int nt);
    void *arg;
};

typedef struct { SllmPool *pool; int tid; } WorkerArg;

static void *worker(void *p)
{
    WorkerArg *wa = (WorkerArg *)p;
    SllmPool *pl = wa->pool;
    int mytid = wa->tid;
    int seen = 0;                     /* last generation this worker ran */
    for (;;) {
        pthread_mutex_lock(&pl->mu);
        /* Wait on `seen`, not on a snapshot of `generation`: taking the
           snapshot at loop entry loses the wakeup if the main thread bumps
           the generation between our `arrive++` and this lock. */
        while (pl->generation == seen && !pl->shutdown)
            pthread_cond_wait(&pl->cv_go, &pl->mu);
        if (pl->shutdown) { pthread_mutex_unlock(&pl->mu); break; }
        int g = pl->generation;
        void (*fn)(void *, int, int) = pl->fn;
        void *arg = pl->arg;
        int nt = pl->n_threads;
        pthread_mutex_unlock(&pl->mu);

        fn(arg, mytid, nt);

        pthread_mutex_lock(&pl->mu);
        pl->arrive++;
        pthread_cond_signal(&pl->cv_done);
        pthread_mutex_unlock(&pl->mu);
        seen = g;
    }
    return NULL;
}

SllmPool *pool_create(int n_threads)
{
    if (n_threads < 1) n_threads = 1;
    if (n_threads > SLLM_MAX_THREADS) n_threads = SLLM_MAX_THREADS;
    SllmPool *p = (SllmPool *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n_threads = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_go, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    for (int i = 1; i < n_threads; i++) {
        WorkerArg *wa = (WorkerArg *)calloc(1, sizeof(WorkerArg));
        wa->pool = p;
        wa->tid = i;
        p->wargs[i] = wa;
        if (pthread_create(&p->th[i], NULL, worker, wa) != 0) {
            p->n_threads = i;
            break;
        }
    }
    return p;
}

void pool_run(SllmPool *p, void (*fn)(void *arg, int tid, int nt), void *arg)
{
    if (!p || p->n_threads <= 1) { fn(arg, 0, 1); return; }
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->arg = arg;
    p->arrive = 0;
    p->generation++;
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mu);

    fn(arg, 0, p->n_threads);

    pthread_mutex_lock(&p->mu);
    while (p->arrive < p->n_threads - 1)
        pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void pool_free(SllmPool *p)
{
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_go);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n_threads; i++) pthread_join(p->th[i], NULL);
    for (int i = 1; i < SLLM_MAX_THREADS; i++) free(p->wargs[i]);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_go);
    pthread_cond_destroy(&p->cv_done);
    free(p);
}

/* ------------------------------------------------------------------- qmv */
typedef struct {
    const uint8_t  *w;
    const float    *sf;
    const uint16_t *sb;
    const int8_t   *x;
    float           xscale;
    int             rows, cols, quant, group;
    float          *out;
} QmvArg;

/* ------------------------------------------------------------ int4 kernel */
/* Weight layout per 32-value sub-block: 16 bytes, byte[i] holds element i in
   its low nibble and element 16+i in its high nibble.  Nibbles are signed
   two's complement (-8..7), so the sign extension is two shifts. */
int32_t dot_i4_i16(const uint8_t *qs, const int16_t *x, int nsub)
{
#if SLLM_NEON
    int32x4_t acc = vdupq_n_s32(0);
    for (int s = 0; s < nsub; s++) {
        uint8x16_t p = vld1q_u8(qs + s * 16);
        /* low nibble -> signed byte: shl 4 puts the nibble's sign bit in bit7,
           the arithmetic shift back sign-extends it */
        int8x16_t lo = vshrq_n_s8(vshlq_n_s8(vreinterpretq_s8_u8(p), 4), 4);
        int8x16_t hi = vshrq_n_s8(vreinterpretq_s8_u8(p), 4);
        int16x8_t l0 = vmovl_s8(vget_low_s8(lo));
        int16x8_t l1 = vmovl_s8(vget_high_s8(lo));
        int16x8_t h0 = vmovl_s8(vget_low_s8(hi));
        int16x8_t h1 = vmovl_s8(vget_high_s8(hi));
        const int16_t *xx = x + s * 32;
        acc = vmlal_s16(acc, vget_low_s16(l0),  vld1_s16(xx));
        acc = vmlal_s16(acc, vget_high_s16(l0), vld1_s16(xx + 4));
        acc = vmlal_s16(acc, vget_low_s16(l1),  vld1_s16(xx + 8));
        acc = vmlal_s16(acc, vget_high_s16(l1), vld1_s16(xx + 12));
        acc = vmlal_s16(acc, vget_low_s16(h0),  vld1_s16(xx + 16));
        acc = vmlal_s16(acc, vget_high_s16(h0), vld1_s16(xx + 20));
        acc = vmlal_s16(acc, vget_low_s16(h1),  vld1_s16(xx + 24));
        acc = vmlal_s16(acc, vget_high_s16(h1), vld1_s16(xx + 28));
    }
#if defined(__aarch64__)
    return vaddvq_s32(acc);
#else
    int32x2_t t = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    t = vpadd_s32(t, t);
    return vget_lane_s32(t, 0);
#endif
#else
    int32_t total = 0;
    for (int s = 0; s < nsub; s++) {
        const uint8_t *q = qs + s * 16;
        const int16_t *xx = x + s * 32;
        for (int i = 0; i < 32; i++) {
            int v = (i < 16) ? (q[i] & 0x0F) : (q[i - 16] >> 4);
            if (v >= 8) v -= 16;
            total += v * (int32_t)xx[i];
        }
    }
    return total;
#endif
}

/* ------------------------------------------------------------ int8 kernel */
static void qmv_worker(void *p, int tid, int nt)
{
    QmvArg *a = (QmvArg *)p;
    int r0 = a->rows * tid / nt;
    int r1 = a->rows * (tid + 1) / nt;

    if (a->quant == Q_I4) {
        int rowb = a->cols / 2;                 /* packed bytes per row   */
        int ng   = a->cols / a->group;          /* scale groups per row    */
        int sub  = a->group / 32;               /* 32-value sub-blocks     */
        int gb   = a->group / 2;                /* packed bytes per group  */
        for (int r = r0; r < r1; r++) {
            const uint8_t  *w  = a->w  + (size_t)r * rowb;
            const uint16_t *sb = a->sb + (size_t)r * ng;
            float acc = 0.0f;
            for (int g = 0; g < ng; g++) {
                int32_t s = dot_i4_i8(w + (size_t)g * gb, a->x + (size_t)g * a->group, sub);
                acc += (float)s * sllm_bf16(sb[g]);
            }
            a->out[r] = acc * a->xscale;
        }
    } else {
        for (int r = r0; r < r1; r++) {
            int32_t s = dot_i8_i8((const int8_t *)a->w + (size_t)r * a->cols,
                                  a->x, a->cols);
            a->out[r] = (float)s * a->sf[r] * a->xscale;
        }
    }
}

void qmv_mat(SllmModel *m, const SllmMat *mat, const int8_t *x, float xscale,
             float *out)
{
    QmvArg a;
    a.w = mat->w; a.sf = mat->sf; a.sb = mat->sb;
    a.x = x; a.xscale = xscale;
    a.rows = mat->rows; a.cols = mat->cols;
    a.quant = mat->quant; a.group = mat->group;
    a.out = out;
    pool_run(m->pool, qmv_worker, &a);
    m->flops += (uint64_t)mat->rows * mat->cols * 2;
}

/* expand one row of W into `out[cols]` floats */
void mat_row_f32(const SllmMat *mat, int row, float *out)
{
    int cols = mat->cols;
    if (mat->quant == Q_I4) {
        const uint8_t *q = mat->w + (size_t)row * (cols / 2);
        int ng = cols / mat->group;
        int gb = mat->group / 2;
        for (int g = 0; g < ng; g++) {
            float sc = sllm_bf16(mat->sb[(size_t)row * ng + g]);
            const uint8_t *p = q + (size_t)g * gb;
            int base = g * mat->group;
            for (int s = 0; s < mat->group / 32; s++) {
                const uint8_t *pp = p + s * 16;
                float *o = out + base + s * 32;
                for (int i = 0; i < 32; i++) {
                    int v = (i < 16) ? (pp[i] & 0x0F) : (pp[i - 16] >> 4);
                    if (v >= 8) v -= 16;
                    o[i] = (float)v * sc;
                }
            }
        }
    } else {
        const int8_t *w = (const int8_t *)mat->w + (size_t)row * cols;
        float sc = mat->sf[row];
        for (int i = 0; i < cols; i++) out[i] = (float)w[i] * sc;
    }
}
