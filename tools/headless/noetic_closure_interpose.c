/* DYLD __interpose open-log for unsigned / adhoc-signed binaries.
 *
 * Two-level namespace: calls to open/stat/etc from THIS dylib bind to
 * libsystem at link time, so the replacements do not recurse. The target
 * image's lookups are redirected. SIP blocks this on Apple-signed binaries;
 * the hawking decode example is adhoc linker-signed and accepts it.
 *
 * OPENLOG_PATH: file to append one "OP PATH" line per call.
 */
#define _DARWIN_C_SOURCE
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/attr.h>
#include <sys/stat.h>
#include <unistd.h>

static int logfd = -1;

static void init(void) __attribute__((constructor));
static void init(void) {
    const char *p = getenv("OPENLOG_PATH");
    if (p) {
        logfd = open(p, O_WRONLY | O_CREAT | O_APPEND, 0644);
    }
    if (logfd >= 0) {
        dprintf(logfd, "CTOR pid=%d\n", getpid());
    }
}

static void logp(const char *op, const char *path) {
    if (logfd >= 0 && path) {
        dprintf(logfd, "%s %s\n", op, path);
    }
}

int my_open(const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    logp("open", path);
    if (flags & O_CREAT) {
        return open(path, flags, mode);
    }
    return open(path, flags);
}

int my_openat(int fd, const char *path, int flags, ...) {
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    logp("openat", path);
    if (flags & O_CREAT) {
        return openat(fd, path, flags, mode);
    }
    return openat(fd, path, flags);
}

int my_stat(const char *path, struct stat *buf) {
    logp("stat", path);
    return stat(path, buf);
}

int my_lstat(const char *path, struct stat *buf) {
    logp("lstat", path);
    return lstat(path, buf);
}

int my_access(const char *path, int amode) {
    logp("access", path);
    return access(path, amode);
}

int my_getattrlist(
    const char *path,
    struct attrlist *attrList,
    void *attrBuf,
    size_t attrBufSize,
    unsigned int options
) {
    logp("getattrlist", path);
    return getattrlist(path, attrList, attrBuf, attrBufSize, options);
}

FILE *my_fopen(const char *path, const char *mode) {
    logp("fopen", path);
    return fopen(path, mode);
}

int my_fstatat(int fd, const char *path, struct stat *buf, int flag) {
    logp("fstatat", path);
    return fstatat(fd, path, buf, flag);
}

#define DYLD_INTERPOSE(_r, _e) \
    __attribute__((used)) static struct { \
        const void *replacement; \
        const void *replacee; \
    } _interpose_##_e __attribute__((section("__DATA,__interpose"))) = { \
        (const void *)(unsigned long)&_r, \
        (const void *)(unsigned long)&_e, \
    }

DYLD_INTERPOSE(my_open, open);
DYLD_INTERPOSE(my_openat, openat);
DYLD_INTERPOSE(my_stat, stat);
DYLD_INTERPOSE(my_lstat, lstat);
DYLD_INTERPOSE(my_access, access);
DYLD_INTERPOSE(my_getattrlist, getattrlist);
DYLD_INTERPOSE(my_fopen, fopen);
DYLD_INTERPOSE(my_fstatat, fstatat);
