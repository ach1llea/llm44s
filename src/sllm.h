/* sllm.h - minimal int8 Transformer (Llama arch) inference engine for tiny devices */
#ifndef SLLM_H
#define SLLM_H

#include <stdint.h>
#include <stddef.h>

#define SLLM_MAX_THREADS 4

/* tensor ids: token embedding, then 9 per layer, then final norm */
enum {
    T_TOK_EMB = 0,
    L_ATTN_NORM = 0, L_WQ, L_WK, L_WV, L_WO, L_FFN_NORM, L_W1, L_W3, L_W2,
    L_N_PER_LAYER = 9
};

typedef struct {
    int32_t  id;
    int32_t  ndim;
    int32_t  shape[4];
    uint32_t nbytes;
    uint32_t off;      /* data offset in file (files are < 4GB) */
    uint32_t soff;     /* scale offset (quantized tensors), 0 if none */
} SllmTensor;

/* weight encodings */
enum {
    Q_NONE = 0,   /* raw fp32                    */
    Q_I8   = 1,   /* int8, one fp32 scale / row  */
    Q_I4   = 2    /* int4 packed, bf16 scale / group of `qgroup` */
};

typedef struct {
    int   dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim;
    int   vocab_size, max_seq;
    float rms_eps, rope_theta;
    int   quant;
    int   qgroup;      /* int4 group size, 0 when not Q_I4 */
} SllmConfig;

/* A quantized matrix.  `w` is rows*cols int8 for Q_I8, or rows*(cols/2)
   packed nibbles for Q_I4 (two values per byte, low nibble = element i,
   high nibble = element i+16 within each 32-value sub-block). */
typedef struct {
    const uint8_t  *w;
    const float    *sf;    /* Q_I8: rows fp32 scales */
    const uint16_t *sb;    /* Q_I4: rows*(cols/qgroup) bf16 scales */
    int rows, cols;
    int quant, group;
} SllmMat;

typedef struct SllmPool SllmPool;

typedef struct {
    SllmConfig  cfg;
    uint8_t    *base;
    size_t      size;
    int         n_tensors;
    int32_t    *tmap;        /* id -> entry slot */
    SllmTensor *tensors;

    float *cos_tab, *sin_tab;     /* [max_seq][head_dim/2] */
    float *kcache, *vcache;       /* [n_layers][max_seq][kv_heads*head_dim] */
    float *x, *xb, *att, *tmp;    /* [dim] / [max(dim,hidden_dim)] */
    float *gate, *up;             /* [hidden_dim] */
    float *kvb;                   /* [kv_heads*head_dim] scratch */
    float *scores;                /* [max_seq] */
    int8_t  *xq;                  /* [max(dim,hidden_dim)] activations, +-127 */
    float *logits;                /* [vocab_size] */

    int      pos;
    SllmPool *pool;
    int      n_threads;
    uint64_t flops;
} SllmModel;

/* logging + process metrics (optional, used by the iOS app UI) */
typedef void (*sllm_log_fn)(const char *line);
void   sllm_set_log(sllm_log_fn fn);
void   sllm_log(const char *fmt, ...);
double sllm_now_ms(void);
long   sllm_rss_kb(void);
long   sllm_free_mb(void);      /* free + inactive, i.e. what a malloc can still get */
long   sllm_mem_used_mb(void);  /* total - free_mb */
long   sllm_mem_total_mb(void);
int    sllm_cpu_percent(void);  /* whole-device CPU load, call at a fixed cadence */

/* thread pool */
SllmPool *pool_create(int n_threads);
void      pool_run(SllmPool *p, void (*fn)(void *arg, int tid, int nt), void *arg);
void      pool_free(SllmPool *p);

/* quantized matvec: out[rows] = W[rows,cols] * x, with int8 or int4 W and
   int8 activations (both in +-127, so vpadalq can accumulate 32-bit sums) */
void  mat_get(SllmModel *m, int tid, SllmMat *mat);
void  qmv_mat(SllmModel *m, const SllmMat *mat, const int8_t *x, float xscale,
              float *out);
/* dequantize a single row of W into out[cols] floats */
void  mat_row_f32(const SllmMat *mat, int row, float *out);
float quantize_act(int8_t *dst, const float *src, int n);
int32_t dot_i8_i8(const int8_t *w, const int8_t *x, int n);
int32_t dot_i4_i8(const uint8_t *w, const int8_t *x, int n32sub);
static inline float sllm_bf16(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}

/* model */
typedef struct {
    int  dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim;
    int  vocab_size, max_seq, quant, qgroup;
    int  mixed;        /* file contains both int4 and int8 tensors */
    long file_size;
    long n_params;
} SllmProbe;

/* read only the header, so the model picker can list files without loading */
int    sllm_probe(const char *path, SllmProbe *out);
int    sllm_load(SllmModel *m, const char *path, int n_threads);
void   sllm_free(SllmModel *m);
void   sllm_reset(SllmModel *m);
const float *sllm_forward(SllmModel *m, int token);
const float *sllm_tensor_f32(SllmModel *m, int tid, int *n);

/* tokenizer */
typedef struct {
    int   n_vocab, n_merges;
    int32_t byte_id[256];
    uint32_t *mkeys, *mvals;
    int   special[3];
    uint8_t *mbase;       /* whole file buffer */
    uint8_t **tok;        /* raw bytes per id */
    uint16_t *toklen;
    int32_t *hash;        /* open-addressed merge table */
    int    hsize;
} SllmTok;

int  tok_load(SllmTok *t, const char *path);
void tok_free(SllmTok *t);
int  tok_encode(SllmTok *t, const char *text, int *ids, int max_ids);
int  tok_piece(SllmTok *t, int id, const uint8_t **bytes);

/* sampler */
typedef struct {
    int      vocab;
    float    temp, top_p, rep_pen;
    int      top_k;
    uint64_t rng;
} SllmSampler;

void sampler_init(SllmSampler *s, int vocab);
int  sampler_pick(SllmSampler *s, float *logits, const int *recent, int n_recent);

/* generation */
typedef void (*token_cb)(void *ud, const char *bytes, int len);
/* prefills prompt[prefill_from..n_prompt) then decodes; does NOT reset the
   model, so callers can keep the KV cache across turns. */
int sllm_generate(SllmModel *m, SllmTok *t, SllmSampler *s,
                  const int *prompt, int n_prompt, int prefill_from,
                  int max_new, token_cb cb, void *ud, int *out_ids, int out_cap);

/* http */
int http_serve(SllmModel *m, SllmTok *t, int port, const char *webdir,
               const char *system_prompt);

#endif
