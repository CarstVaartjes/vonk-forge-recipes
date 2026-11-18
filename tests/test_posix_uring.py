"""The POSIX liburing stand-in reads through pread and never calls io_uring."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "tools/posix-uring"

HARNESS = r"""
#include <fcntl.h>
#include <stdio.h>
#include <liburing.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "failed: %s line %d\n", #c, __LINE__); return 1; } } while (0)

static int read_one(struct io_uring *ring, int fd, void *buf, unsigned len, uint64_t off,
                    int slot, int fixed, int *res)
{
    struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
    if (!sqe) return -1;
    io_uring_prep_read_fixed(sqe, fd, buf, len, off, slot);
    if (fixed) sqe->flags |= IOSQE_FIXED_FILE;
    io_uring_sqe_set_data64(sqe, 7);
    if (io_uring_submit(ring) != 1) return -1;
    struct io_uring_cqe *cqe = NULL;
    if (io_uring_wait_cqe(ring, &cqe) != 0 || io_uring_cqe_get_data64(cqe) != 7) return -1;
    *res = cqe->res;
    io_uring_cqe_seen(ring, cqe);
    return 0;
}

int main(int argc, char **argv)
{
    (void)argc;
    int fd = open(argv[1], O_RDONLY);
    CHECK(fd >= 0);
    static char buffer[8192];
    struct iovec slots[2] = {{buffer, 4096}, {buffer + 4096, 4096}};
    struct io_uring ring;
    CHECK(io_uring_queue_init(4, &ring, 0) == 0);
    CHECK(io_uring_register_buffers(&ring, slots, 2) == 0);
    CHECK(io_uring_register_files(&ring, &fd, 1) == 0);
    uint64_t high = (1ULL << 32) + 3;
    int res = 0;
    /* unaligned read above 4 GiB; the file holds "ABCDEFGH" at that offset */
    CHECK(read_one(&ring, 0, buffer, 4, high, 0, 1, &res) == 0 && res == 4);
    CHECK(memcmp(buffer, "ABCD", 4) == 0);
    /* short read at the end of the file, then nothing past it */
    CHECK(read_one(&ring, 0, buffer, 100, high + 4, 0, 1, &res) == 0 && res == 4);
    CHECK(memcmp(buffer, "EFGH", 4) == 0);
    CHECK(read_one(&ring, 0, buffer, 16, high + 8, 0, 1, &res) == 0 && res == 0);
    /* an unregistered fixed index fails with EBADF; a plain fd needs no registration */
    CHECK(read_one(&ring, 5, buffer, 4, high, 0, 1, &res) == 0 && res == -EBADF);
    CHECK(read_one(&ring, fd, buffer, 4, high, 0, 0, &res) == 0 && res == 4);
    /* a read outside its registered buffer fails with EFAULT */
    CHECK(read_one(&ring, 0, buffer, 5000, high, 0, 1, &res) == 0 && res == -EFAULT);
    /* two reads drain through for_each_cqe / cq_advance */
    for (unsigned i = 0; i < 2; i++) {
        struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
        CHECK(sqe);
        io_uring_prep_read_fixed(sqe, 0, buffer + 4096 * i, 4, high + 4 * i, (int)i);
        sqe->flags |= IOSQE_FIXED_FILE;
        io_uring_sqe_set_data64(sqe, i);
    }
    CHECK(io_uring_submit(&ring) == 2);
    unsigned head, seen = 0;
    struct io_uring_cqe *cqe;
    io_uring_for_each_cqe(&ring, head, cqe) {
        CHECK(cqe->res == 4 && io_uring_cqe_get_data64(cqe) == seen);
        seen++;
    }
    CHECK(seen == 2);
    io_uring_cq_advance(&ring, seen);
    CHECK(io_uring_wait_cqe(&ring, &cqe) == -EAGAIN);
    CHECK(memcmp(buffer, "ABCD", 4) == 0 && memcmp(buffer + 4096, "EFGH", 4) == 0);
    CHECK(io_uring_unregister_files(&ring) == 0);
    io_uring_queue_exit(&ring);
    return 0;
}
"""


def test_shim_serves_fixed_reads_with_pread(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    assert compiler, "the native shim gate requires a C compiler"
    source = tmp_path / "harness.c"
    source.write_text(HARNESS)
    binary = tmp_path / "harness"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-D_POSIX_C_SOURCE=200809L",
            "-D_FILE_OFFSET_BITS=64",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{SHIM}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    data = tmp_path / "sparse.bin"
    with data.open("wb") as handle:
        handle.seek((1 << 32) + 3)
        handle.write(b"ABCDEFGH")
    subprocess.run([str(binary), str(data)], check=True)


def test_shim_makes_no_io_uring_call() -> None:
    header = (SHIM / "liburing.h").read_text()
    for forbidden in ("syscall(", "__NR_io_uring", "linux/io_uring.h"):
        assert forbidden not in header


def test_adapter_copies_are_byte_identical() -> None:
    for name in ("liburing.h", "liburing.pc"):
        canonical = (SHIM / name).read_bytes()
        for copy in sorted((ROOT / "adapters").glob(f"**/posix-uring/{name}")):
            assert copy.read_bytes() == canonical, (
                f"{copy} differs from tools/posix-uring"
            )
