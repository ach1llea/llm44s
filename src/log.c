/* log.c - optional log sink + a couple of process metrics for the app UI */
#include "sllm.h"

#include <mach/mach.h>
#include <mach/processor_info.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <sys/time.h>

static sllm_log_fn g_log = NULL;

void sllm_set_log(sllm_log_fn fn) { g_log = fn; }

void sllm_log(const char *fmt, ...)
{
    char buf[512];
    va_list ap;
    if (!g_log) return;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_log(buf);
}

double sllm_now_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

long sllm_rss_kb(void)
{
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return -1;
#ifdef __APPLE__
    return ru.ru_maxrss / 1024;
#else
    return ru.ru_maxrss;
#endif
}

long sllm_free_mb(void)
{
    mach_port_t host = mach_host_self();
    vm_size_t page = 0;
    vm_statistics_data_t st;
    mach_msg_type_number_t cnt = HOST_VM_INFO_COUNT;
    if (host_page_size(host, &page) != KERN_SUCCESS) return -1;
    if (host_statistics(host, HOST_VM_INFO, (host_info_t)&st, &cnt) != KERN_SUCCESS) return -1;
    return (long)((double)(st.free_count + st.inactive_count) * (double)page / (1024.0 * 1024.0));
}

long sllm_mem_total_mb(void)
{
    long long bytes = 0;
    size_t len = sizeof(bytes);
    /* sysctlbyname is simpler than host_statistics for the physical size, and
       hw.memsize is a plain integer on every iOS version back to 2.x */
    if (sysctlbyname("hw.memsize", &bytes, &len, NULL, 0) != 0 || bytes <= 0) {
        /* 4S reports 504 MB of usable DRAM, but never guess a wrong number here:
           fall back to what the VM counters can tell us */
        return -1;
    }
    return (long)(bytes / (1024 * 1024));
}

long sllm_mem_used_mb(void)
{
    long total = sllm_mem_total_mb();
    long avail = sllm_free_mb();
    if (total <= 0 || avail < 0) return -1;
    long used = total - avail;
    return used < 0 ? 0 : used;
}

int sllm_cpu_percent(void)
{
    /* CPU load from the per-processor tick counters.  A single sample is
       meaningless (it is a cumulative count), so the first call only records a
       baseline.  Call it at a fixed cadence and read the delta.
       The host port is cached: mach_host_self() hands back a new send right on
       every call, and at 2 Hz that would eventually run the task out of ports. */
    static mach_port_t host;
    if (!host) host = mach_host_self();

    natural_t ncpu = 0;
    processor_info_array_t info = NULL;
    mach_msg_type_number_t n = 0;
    if (host_processor_info(host, PROCESSOR_CPU_LOAD_INFO,
                            &ncpu, &info, &n) != KERN_SUCCESS || !info) {
        /* Some builds refuse host-wide counters; fall back to how much CPU this
           process itself burned, which is what the number is really about. */
        struct rusage ru;
        struct timeval tv;
        static long long p_us;
        static double p_wall;
        if (getrusage(RUSAGE_SELF, &ru) != 0 || gettimeofday(&tv, NULL) != 0)
            return -1;
        long long us = (long long)ru.ru_utime.tv_sec * 1000000 + ru.ru_utime.tv_usec
                     + (long long)ru.ru_stime.tv_sec * 1000000 + ru.ru_stime.tv_usec;
        double wall = (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
        double dw = wall - p_wall;
        long long du = us - p_us;
        p_us = us; p_wall = wall;
        if (dw <= 0.0) return 0;
        int pct = (int)((double)du / (dw * 1000.0) * 100.0);
        return pct < 0 ? 0 : (pct > 200 ? 200 : pct);
    }

    uint64_t total = 0, idle = 0;
    for (natural_t i = 0; i < ncpu; i++) {
        natural_t b = i * CPU_STATE_MAX;
        total += (uint64_t)info[b + CPU_STATE_USER]
               + (uint64_t)info[b + CPU_STATE_SYSTEM]
               + (uint64_t)info[b + CPU_STATE_NICE]
               + (uint64_t)info[b + CPU_STATE_IDLE];
        idle  += (uint64_t)info[b + CPU_STATE_IDLE];
    }
    vm_deallocate(mach_task_self(), (vm_address_t)info,
                  (vm_size_t)n * sizeof(integer_t));

    static uint64_t p_total, p_idle;
    static int primed;
    if (!primed) {
        p_total = total; p_idle = idle; primed = 1;
        return (int)((total - idle) * 100ULL / (total ? total : 1));
    }
    uint64_t dt = total - p_total, di = idle - p_idle;
    p_total = total; p_idle = idle;
    if (dt == 0) return 0;
    if (di > dt) di = dt;
    return (int)((dt - di) * 100ULL / dt);
}
