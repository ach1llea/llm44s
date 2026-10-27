/* http.c - tiny HTTP/1.1 + SSE server, one request at a time (personal use) */
#include "sllm.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

static double now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static void send_all(int fd, const char *s, int n)
{
    int off = 0;
    while (off < n) {
        ssize_t w = write(fd, s + off, (size_t)(n - off));
        if (w <= 0) return;
        off += (int)w;
    }
}

static void send_str(int fd, const char *s) { send_all(fd, s, (int)strlen(s)); }

static int read_request(int fd, char *buf, int cap)
{
    int n = 0;
    while (n < cap - 1) {
        int r = (int)read(fd, buf + n, (size_t)(cap - 1 - n));
        if (r <= 0) break;
        n += r;
        buf[n] = 0;
        if (strstr(buf, "\r\n\r\n") || strstr(buf, "\n\n")) break;
    }
    return n;
}

static int hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void url_decode(const char *src, char *dst, int cap)
{
    int i = 0, o = 0;
    while (src[i] && o < cap - 1) {
        if (src[i] == '%' && hexval(src[i + 1]) >= 0 && hexval(src[i + 2]) >= 0) {
            dst[o++] = (char)(hexval(src[i + 1]) * 16 + hexval(src[i + 2]));
            i += 3;
        } else if (src[i] == '+') {
            dst[o++] = ' ';
            i++;
        } else {
            dst[o++] = src[i++];
        }
    }
    dst[o] = 0;
}

static const char *param(const char *q, const char *key, char *out, int cap)
{
    int klen = (int)strlen(key);
    const char *p = q;
    if (!q) return NULL;
    while (*p) {
        if (strncmp(p, key, (size_t)klen) == 0 && p[klen] == '=') {
            const char *v = p + klen + 1;
            const char *e = strchr(v, '&');
            int len = e ? (int)(e - v) : (int)strlen(v);
            if (len >= cap) len = cap - 1;
            memcpy(out, v, (size_t)len);
            out[len] = 0;
            return out;
        }
        p = strchr(p, '&');
        if (!p) break;
        p++;
    }
    return NULL;
}

/* ------------------------------------------------------------ conversation */
typedef struct {
    SllmTok *tok;
    char     sys[480];
    int     *ids;
    int      n;
    int      cap;
    int      base_n;
    int      kv;         /* tokens currently in the model's KV cache */
    float    last_tok_s;
} Conv;

static void conv_push_id(Conv *c, int id)
{
    if (c->n < c->cap) c->ids[c->n++] = id;
}

static void conv_push_text(Conv *c, const char *s)
{
    if (c->n < c->cap)
        c->n += tok_encode(c->tok, s, c->ids + c->n, c->cap - c->n);
}

static void conv_push_ids(Conv *c, const int *ids, int n)
{
    for (int i = 0; i < n && c->n < c->cap; i++) c->ids[c->n++] = ids[i];
}

static void conv_build_system(Conv *c)
{
    c->n = 0;
    conv_push_id(c, c->tok->special[1]);
    conv_push_text(c, "system\n");
    conv_push_text(c, c->sys);
    conv_push_id(c, c->tok->special[2]);
    conv_push_text(c, "\n");
    c->base_n = c->n;
}

/* --------------------------------------------------------------- streaming */
typedef struct {
    int  fd;
    char pend[16];
    int  npend;
    int  count;
} Stream;

static int utf8_safe_len(const char *buf, int n)
{
    int i = n - 1, back = 0;
    if (n <= 0) return 0;
    while (i >= 0 && ((unsigned char)buf[i] & 0xC0) == 0x80 && back < 3) { i--; back++; }
    if (i < 0) return 0;
    unsigned char c = (unsigned char)buf[i];
    int need = (c & 0x80) == 0 ? 1 : ((c & 0xE0) == 0xC0 ? 2 : ((c & 0xF0) == 0xE0 ? 3 : 4));
    return (n - i >= need) ? n : i;
}

static void sse_emit(Stream *st, const char *bytes, int len)
{
    char esc[128];
    int o = 0;
    for (int i = 0; i < len && o < (int)sizeof(esc) - 8; i++) {
        unsigned char ch = (unsigned char)bytes[i];
        if (ch == '"' || ch == '\\') { esc[o++] = '\\'; esc[o++] = (char)ch; }
        else if (ch == '\n') { esc[o++] = '\\'; esc[o++] = 'n'; }
        else if (ch == '\r') { esc[o++] = '\\'; esc[o++] = 'r'; }
        else if (ch < 0x20) { continue; }
        else esc[o++] = (char)ch;
    }
    esc[o] = 0;
    send_str(st->fd, "data: {\"t\":\"");
    send_all(st->fd, esc, o);
    send_str(st->fd, "\"}\n\n");
}

static void on_token(void *ud, const char *bytes, int len)
{
    Stream *st = (Stream *)ud;
    if (len <= 0) return;
    if (st->npend + len > (int)sizeof(st->pend)) {
        sse_emit(st, st->pend, st->npend);
        st->npend = 0;
    }
    if (len > (int)sizeof(st->pend)) len = (int)sizeof(st->pend);
    memcpy(st->pend + st->npend, bytes, (size_t)len);
    st->npend += len;
    int safe = utf8_safe_len(st->pend, st->npend);
    if (safe > 0) {
        sse_emit(st, st->pend, safe);
        memmove(st->pend, st->pend + safe, (size_t)(st->npend - safe));
        st->npend -= safe;
    }
    st->count++;
}

/* ---------------------------------------------------------------- handlers */
static void serve_file(int fd, const char *dir, const char *name, const char *ctype)
{
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        send_str(fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 26\r\nConnection: close\r\n\r\nindex.html not found here\n");
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *body = (char *)malloc((size_t)sz + 1);
    if (!body || fread(body, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(body); return; }
    fclose(f);
    char hdr[256];
    int hl = snprintf(hdr, sizeof(hdr),
                      "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nContent-Length: %ld\r\n"
                      "Cache-Control: no-store\r\nConnection: close\r\n\r\n", ctype, sz);
    send_all(fd, hdr, hl);
    send_all(fd, body, (int)sz);
    free(body);
}

static void send_json(int fd, const char *body)
{
    char hdr[160];
    int hl = snprintf(hdr, sizeof(hdr),
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n"
        "Connection: close\r\n\r\n", (int)strlen(body));
    send_all(fd, hdr, hl);
    send_str(fd, body);
}

int http_serve(SllmModel *m, SllmTok *t, int port, const char *webdir,
               const char *system_prompt)
{
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    if (ls < 0) { perror("socket"); return -1; }
    int one = 1;
    setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(sa));
    sa.sin_family = AF_INET;
    sa.sin_port = htons((unsigned short)port);
    sa.sin_addr.s_addr = inet_addr("127.0.0.1");
    if (bind(ls, (struct sockaddr *)&sa, sizeof(sa)) != 0) { perror("bind"); return -1; }
    if (listen(ls, 4) != 0) { perror("listen"); return -1; }
    printf("serving on http://127.0.0.1:%d  ctx=%d threads=%d\n",
           port, m->cfg.max_seq, m->n_threads);
    fflush(stdout);

    Conv conv;
    memset(&conv, 0, sizeof(conv));
    conv.tok = t;
    conv.cap = m->cfg.max_seq > 64 ? m->cfg.max_seq : 64;
    conv.ids = (int *)calloc((size_t)conv.cap, sizeof(int));
    snprintf(conv.sys, sizeof(conv.sys), "%s",
             system_prompt ? system_prompt : "You are a helpful assistant");
    conv_build_system(&conv);

    int *gen_ids = (int *)calloc((size_t)conv.cap, sizeof(int));

    SllmSampler smp;
    sampler_init(&smp, m->cfg.vocab_size);

    char req[8192], path[2048], query[4096], val[512], decoded[2048];

    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        int n = read_request(fd, req, (int)sizeof(req));
        if (n <= 0) { close(fd); continue; }

        path[0] = query[0] = 0;
        if (sscanf(req, "GET %2047s", path) != 1) {
            send_str(fd, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
            close(fd);
            continue;
        }
        char *qm = strchr(path, '?');
        if (qm) { *qm = 0; snprintf(query, sizeof(query), "%s", qm + 1); }

        if (strcmp(path, "/") == 0 || strcmp(path, "/index.html") == 0) {
            serve_file(fd, webdir, "index.html", "text/html; charset=utf-8");
        } else if (strcmp(path, "/favicon.ico") == 0) {
            send_str(fd, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
        } else if (strcmp(path, "/reset") == 0) {
            conv.n = conv.base_n;
            conv.kv = 0;
            sllm_reset(m);
            send_json(fd, "{\"ok\":1}");
        } else if (strcmp(path, "/stats") == 0) {
            char out[256];
            snprintf(out, sizeof(out),
                     "{\"ctx_used\":%d,\"ctx_max\":%d,\"threads\":%d,"
                     "\"tok_s\":%.2f,\"hidden\":%d,\"layers\":%d,\"vocab\":%d}",
                     conv.n, m->cfg.max_seq, m->n_threads, conv.last_tok_s,
                     m->cfg.dim, m->cfg.n_layers, m->cfg.vocab_size);
            send_json(fd, out);
        } else if (strcmp(path, "/chat") == 0) {
            if (!param(query, "q", val, sizeof(val))) {
                send_json(fd, "{\"err\":\"missing q\"}");
                close(fd);
                continue;
            }
            url_decode(val, decoded, sizeof(decoded));

            int max_new = 96;
            if (param(query, "n", val, sizeof(val))) max_new = atoi(val);
            smp.temp = 0.85f;
            smp.top_k = 50;
            smp.top_p = 0.90f;
            smp.rep_pen = 1.05f;
            if (param(query, "temp", val, sizeof(val)))  smp.temp = (float)atof(val);
            if (param(query, "top_p", val, sizeof(val))) smp.top_p = (float)atof(val);
            if (param(query, "top_k", val, sizeof(val))) smp.top_k = atoi(val);
            if (param(query, "rep", val, sizeof(val)))   smp.rep_pen = (float)atof(val);
            if (max_new < 1) max_new = 1;
            if (max_new > conv.cap - 48) max_new = conv.cap - 48;
            if (max_new < 1) max_new = 1;

            /* roll back to the system prefix when the context is nearly full */
            int need = (int)strlen(decoded) + 24;
            if (conv.n + need + max_new + 8 >= conv.cap) {
                conv.n = conv.base_n;
                conv.kv = 0;
                sllm_reset(m);
            }
            int prefill_from = conv.kv;

            conv_push_id(&conv, t->special[1]);
            conv_push_text(&conv, "user\n");
            conv_push_text(&conv, decoded);
            conv_push_id(&conv, t->special[2]);
            conv_push_text(&conv, "\n");
            conv_push_id(&conv, t->special[1]);
            conv_push_text(&conv, "assistant\n");
            int prompt_len = conv.n;

            send_str(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
                         "Cache-Control: no-store\r\nConnection: close\r\n"
                         "X-Accel-Buffering: no\r\n\r\n");

            Stream st;
            memset(&st, 0, sizeof(st));
            st.fd = fd;

            double t0 = now_ms();
            int produced = sllm_generate(m, t, &smp, conv.ids, prompt_len, prefill_from,
                                         max_new, on_token, &st, gen_ids,
                                         conv.cap - prompt_len);
            double dt = now_ms() - t0;
            if (st.npend > 0) { sse_emit(&st, st.pend, st.npend); st.npend = 0; }
            conv.last_tok_s = produced > 0 ? (float)(produced * 1000.0 / (dt > 1.0 ? dt : 1.0)) : 0.0f;

            char out[192];
            int l = snprintf(out, sizeof(out),
                             "data: {\"done\":1,\"n\":%d,\"ms\":%.0f,\"tok_s\":%.2f}\n\n",
                             produced, dt, conv.last_tok_s);
            send_all(fd, out, l);
            close(fd);

            /* keep the assistant turn in context; only the prefill + generated
               tokens are actually in the KV cache, the rest is prefilled next time */
            conv.n = prompt_len;
            conv_push_ids(&conv, gen_ids, produced);
            conv_push_id(&conv, t->special[2]);
            conv_push_text(&conv, "\n");
            conv.kv = m->pos;
            if (conv.kv >= conv.cap - 8) { conv.n = conv.base_n; conv.kv = 0; sllm_reset(m); }
            continue;
        } else {
            send_str(fd, "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nnot found");
        }
        close(fd);
    }
    return 0;
}
