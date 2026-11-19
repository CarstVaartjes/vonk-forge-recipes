/*
 * POSIX stand-in for the liburing subset that B12X, vLLM engram_rows and
 * vLLM direct_io use. submit() runs one pread() per queued read and posts the
 * completion; no io_uring syscall is made, so the default seccomp profile is
 * enough. Completions carry bytes read, 0 at end of file, or -errno.
 * Header only; found through pkg-config as "liburing" (liburing.pc).
 */
#ifndef VONK_POSIX_LIBURING_H
#define VONK_POSIX_LIBURING_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#define IOSQE_FIXED_FILE (1U << 0)

struct io_uring_sqe {
    unsigned flags;
    int fd;
    void *addr;
    unsigned len;
    uint64_t off;
    unsigned buf_index;
    uint64_t user_data;
};

struct io_uring_cqe {
    uint64_t user_data;
    int32_t res;
    uint32_t flags;
};

struct io_uring {
    struct {
        unsigned ring_entries;
        size_t ring_sz;
        void *ring_ptr;
        struct io_uring_sqe *sqes;
        unsigned queued;
    } sq;
    struct {
        unsigned ring_entries;
        size_t ring_sz;
        void *ring_ptr;
        struct io_uring_cqe *cqes;
        unsigned head;
        unsigned tail;
    } cq;
    int *files;
    unsigned nr_files;
    struct iovec *buffers;
    unsigned nr_buffers;
};

static inline int io_uring_queue_init(unsigned entries, struct io_uring *ring, unsigned flags)
{
    (void)flags;
    memset(ring, 0, sizeof(*ring));
    if (entries == 0 || entries > (1U << 15)) return -EINVAL;
    unsigned sq = 1;
    while (sq < entries) sq <<= 1;
    size_t sq_bytes = (size_t)sq * sizeof(struct io_uring_sqe);
    size_t cq_bytes = (size_t)sq * 2 * sizeof(struct io_uring_cqe);
    char *base = (char *)calloc(1, sq_bytes + cq_bytes);
    if (!base) return -ENOMEM;
    ring->sq.ring_entries = sq;
    ring->sq.ring_sz = sq_bytes + cq_bytes;
    ring->sq.ring_ptr = base;
    ring->sq.sqes = (struct io_uring_sqe *)base;
    ring->cq.ring_entries = sq * 2;
    ring->cq.ring_sz = ring->sq.ring_sz;
    ring->cq.ring_ptr = base;
    ring->cq.cqes = (struct io_uring_cqe *)(base + sq_bytes);
    return 0;
}

static inline void io_uring_queue_exit(struct io_uring *ring)
{
    free(ring->sq.ring_ptr);
    free(ring->files);
    free(ring->buffers);
    memset(ring, 0, sizeof(*ring));
}

static inline int io_uring_register_buffers(struct io_uring *ring, const struct iovec *iovecs, unsigned nr)
{
    if (ring->buffers) return -EBUSY;
    if (nr == 0 || !iovecs) return -EINVAL;
    ring->buffers = (struct iovec *)malloc((size_t)nr * sizeof(*iovecs));
    if (!ring->buffers) return -ENOMEM;
    memcpy(ring->buffers, iovecs, (size_t)nr * sizeof(*iovecs));
    ring->nr_buffers = nr;
    return 0;
}

static inline int io_uring_register_files(struct io_uring *ring, const int *files, unsigned nr)
{
    if (ring->files) return -EBUSY;
    if (nr == 0 || !files) return -EINVAL;
    ring->files = (int *)malloc((size_t)nr * sizeof(*files));
    if (!ring->files) return -ENOMEM;
    memcpy(ring->files, files, (size_t)nr * sizeof(*files));
    ring->nr_files = nr;
    return 0;
}

static inline int io_uring_unregister_files(struct io_uring *ring)
{
    if (!ring->files) return -ENXIO;
    free(ring->files);
    ring->files = NULL;
    ring->nr_files = 0;
    return 0;
}

static inline struct io_uring_sqe *io_uring_get_sqe(struct io_uring *ring)
{
    if (ring->sq.queued >= ring->sq.ring_entries) return NULL;
    struct io_uring_sqe *sqe = &ring->sq.sqes[ring->sq.queued++];
    memset(sqe, 0, sizeof(*sqe));
    return sqe;
}

static inline void io_uring_prep_read_fixed(struct io_uring_sqe *sqe, int fd, void *buf,
                                            unsigned nbytes, uint64_t offset, int buf_index)
{
    sqe->fd = fd;
    sqe->addr = buf;
    sqe->len = nbytes;
    sqe->off = offset;
    sqe->buf_index = (unsigned)buf_index;
}

static inline void io_uring_sqe_set_data64(struct io_uring_sqe *sqe, uint64_t data)
{
    sqe->user_data = data;
}

static inline uint64_t io_uring_cqe_get_data64(const struct io_uring_cqe *cqe)
{
    return cqe->user_data;
}

static inline int io_uring_posix_read(const struct io_uring *ring, const struct io_uring_sqe *sqe)
{
    int fd = sqe->fd;
    if (sqe->flags & IOSQE_FIXED_FILE) {
        if (fd < 0 || (unsigned)fd >= ring->nr_files) return -EBADF;
        fd = ring->files[fd];
    }
    if (fd < 0) return -EBADF;
    if (sqe->buf_index >= ring->nr_buffers) return -EFAULT;
    const struct iovec *slot = &ring->buffers[sqe->buf_index];
    const char *base = (const char *)slot->iov_base;
    const char *at = (const char *)sqe->addr;
    if (at < base || (size_t)(at - base) > slot->iov_len || sqe->len > slot->iov_len - (size_t)(at - base))
        return -EFAULT;
    ssize_t n;
    do {
        n = pread(fd, sqe->addr, sqe->len, (off_t)sqe->off);
    } while (n < 0 && errno == EINTR);
    return n < 0 ? -errno : (int)n;
}

/* Runs every queued read now and returns how many were submitted. */
static inline int io_uring_submit(struct io_uring *ring)
{
    unsigned total = ring->sq.queued;
    unsigned done = 0;
    while (done < total && ring->cq.tail - ring->cq.head < ring->cq.ring_entries) {
        const struct io_uring_sqe *sqe = &ring->sq.sqes[done];
        struct io_uring_cqe *cqe = &ring->cq.cqes[ring->cq.tail & (ring->cq.ring_entries - 1)];
        cqe->user_data = sqe->user_data;
        cqe->res = io_uring_posix_read(ring, sqe);
        cqe->flags = 0;
        ring->cq.tail++;
        done++;
    }
    if (total && !done) return -EBUSY;
    memmove(ring->sq.sqes, ring->sq.sqes + done, (size_t)(total - done) * sizeof(*ring->sq.sqes));
    ring->sq.queued = total - done;
    return (int)done;
}

/* Nothing is ever in flight, so an empty completion queue cannot fill later. */
static inline int io_uring_wait_cqe(struct io_uring *ring, struct io_uring_cqe **cqe_ptr)
{
    if (ring->cq.head == ring->cq.tail) {
        *cqe_ptr = NULL;
        return -EAGAIN;
    }
    *cqe_ptr = &ring->cq.cqes[ring->cq.head & (ring->cq.ring_entries - 1)];
    return 0;
}

static inline void io_uring_cq_advance(struct io_uring *ring, unsigned nr)
{
    ring->cq.head += nr;
}

static inline void io_uring_cqe_seen(struct io_uring *ring, struct io_uring_cqe *cqe)
{
    if (cqe) io_uring_cq_advance(ring, 1);
}

#define io_uring_for_each_cqe(ring, head, cqe)                                            \
    for ((head) = (ring)->cq.head;                                                        \
         ((cqe) = ((head) != (ring)->cq.tail                                              \
                       ? &(ring)->cq.cqes[(head) & ((ring)->cq.ring_entries - 1)] : NULL)); \
         (head)++)

#endif
