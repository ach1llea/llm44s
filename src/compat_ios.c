/* compat.c - memcpy/memset/memmove for the iOS 6 build.
 *
 * The iPhoneOS9.3 SDK's text stubs do not declare _memcpy/_memset/_memmove on
 * any architecture, so ld cannot resolve the libcalls clang emits.  Defining
 * them locally costs nothing (the calls become internal) and lets the binary
 * keep a minimal dylib dependency list.
 *
 * Compiled with -fno-builtin-* so the loops are not turned back into libcalls.
 */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *dst, const void *src, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (((uintptr_t)d | (uintptr_t)s) % sizeof(uint32_t) == 0) {
        size_t w = n / sizeof(uint32_t);
        uint32_t *dw = (uint32_t *)d;
        const uint32_t *sw = (const uint32_t *)s;
        for (size_t i = 0; i < w; i++) dw[i] = sw[i];
        d += w * sizeof(uint32_t);
        s += w * sizeof(uint32_t);
        n -= w * sizeof(uint32_t);
    }
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    unsigned char v = (unsigned char)c;
    while (n && ((uintptr_t)d % sizeof(uint32_t)) != 0) { *d++ = v; n--; }
    if (n >= sizeof(uint32_t)) {
        uint32_t w = (uint32_t)v * 0x01010101u;
        uint32_t *dw = (uint32_t *)d;
        size_t nw = n / sizeof(uint32_t);
        for (size_t i = 0; i < nw; i++) dw[i] = w;
        d += nw * sizeof(uint32_t);
        n -= nw * sizeof(uint32_t);
    }
    while (n--) *d++ = v;
    return dst;
}

void *memmove(void *dst, const void *src, size_t n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d == s || n == 0) return dst;
    if (d < s || d >= s + n) return memcpy(dst, src, n);
    d += n;
    s += n;
    while (n--) *--d = *--s;
    return dst;
}

int memcmp(const void *a, const void *b, size_t n)
{
    const unsigned char *x = (const unsigned char *)a;
    const unsigned char *y = (const unsigned char *)b;
    for (size_t i = 0; i < n; i++) if (x[i] != y[i]) return (int)x[i] - (int)y[i];
    return 0;
}

int strcmp(const char *a, const char *b)
{
    while (*a && *a == *b) { a++; b++; }
    return (int)(unsigned char)*a - (int)(unsigned char)*b;
}

int strncmp(const char *a, const char *b, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return (int)(unsigned char)a[i] - (int)(unsigned char)b[i];
        if (!a[i]) return 0;
    }
    return 0;
}

char *strchr(const char *s, int c)
{
    char ch = (char)c;
    for (;; s++) {
        if (*s == ch) return (char *)s;
        if (!*s) return 0;
    }
}
