"""Execute the image's native row reader without CUDA or liburing installed."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import sysconfig
from array import array
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "adapters/deepseek/0xsero-exl3-vllm-dual/_ple_reader.c"


@pytest.fixture(scope="module")
def reader(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    compiler = shutil.which("cc")
    assert compiler, "the native transport behaviour gate requires a C compiler"
    directory = tmp_path_factory.mktemp("posix-reader")
    harness = directory / "reader.c"
    harness.write_text(
        """
#define _GNU_SOURCE
#include <Python.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
/* macOS has no O_DIRECT. Linux image builds use the real flag. */
#ifndef O_DIRECT
#define O_DIRECT 0
#endif
typedef struct { char message[1024]; } failure_t;
static void system_error(failure_t *f, const char *op) {
    snprintf(f->message, sizeof(f->message), "%s: %s", op, strerror(errno));
}
"""
        + f'#include "{SOURCE}"\n'
        + """
static PyMethodDef methods[] = {
    {"create", py_ple_reader, METH_VARARGS, NULL},
    {"add", py_ple_reader_add, METH_VARARGS, NULL},
    {"run", py_ple_reader_run, METH_VARARGS, NULL},
    {"stats", py_ple_reader_stats, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};
static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, .m_name = "posix_reader", .m_size = -1, .m_methods = methods
};
PyMODINIT_FUNC PyInit_posix_reader(void) { return PyModule_Create(&module); }
"""
    )
    output = directory / ("posix_reader" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        compiler,
        "-std=c99",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-D_FILE_OFFSET_BITS=64",
        f"-I{sysconfig.get_path('include')}",
        str(harness),
        "-o",
        str(output),
    ]
    if sys.platform == "darwin":
        command.extend(["-undefined", "dynamic_lookup"])
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    spec = importlib.util.spec_from_file_location("posix_reader", output)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_rows_scales_offsets_duplicates_and_bounded_staging(
    reader: ModuleType, tmp_path: Path
) -> None:
    # Both planes have unaligned checkpoint offsets and partial EOF blocks.
    weights = bytes(range(251)) * 45
    scales = bytes(range(37)) * 7
    source = tmp_path / "weights"
    source.write_bytes(b"x" * 31 + weights + scales)
    row_bytes, rows = 257, 40
    cache = reader.create(rows, rows, 0, rows, row_bytes, 3, 2048, 64)
    reader.add(cache, 0, str(source), 31, False)
    reader.add(cache, 0, str(source), 31 + len(weights), True)
    with pytest.raises(RuntimeError, match="already registered"):
        reader.add(cache, 0, str(source), 31, False)
    # Repeat enough rows to exercise radix sorting, then reuse the same cache.
    for ids in ([39, 0, 17, 17, -1, 40] * 200, [5, 2, 5]):
        output, scale_output = bytearray(len(ids) * row_bytes), bytearray(len(ids) * 3)
        reader.run(cache, array("q", ids), output, scale_output, len(ids))
        assert output == b"".join(
            weights[i * row_bytes : (i + 1) * row_bytes]
            if 0 <= i < rows
            else bytes(row_bytes)
            for i in ids
        )
        assert scale_output == b"".join(
            scales[i * 3 : (i + 1) * 3] if 0 <= i < rows else bytes(3) for i in ids
        )
        stats = reader.stats(cache)
        assert stats["staging_bytes"] == 65536
        assert stats["read_calls"] < len(ids)


def test_truncated_file_fails_without_returning_partial_rows(
    reader: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "truncated"
    source.write_bytes(b"a" * 8192)
    cache = reader.create(2, 2, 0, 2, 4096, 0, 2, 64)
    reader.add(cache, 0, str(source), 0, False)
    source.write_bytes(b"a" * 4096)
    with pytest.raises(RuntimeError, match="short PLE read"):
        reader.run(cache, array("q", [1]), bytearray(4096), None, 1)


def test_large_checkpoint_offsets_and_rank_scope(
    reader: ModuleType, tmp_path: Path
) -> None:
    source = tmp_path / "large-offset"
    offset = (1 << 32) + 31
    with source.open("wb") as handle:
        handle.seek(offset)
        handle.write(b"abcdefgh")
    cache = reader.create(2, 2, 1, 2, 4, 0, 4, 1)
    reader.add(cache, 0, str(source), offset, False)
    output = bytearray(16)
    reader.run(cache, array("q", [1, 0, -1, 2]), output, None, 4)
    assert output == b"efgh" + bytes(12)


def test_transport_cannot_call_uring_and_image_installs_it() -> None:
    source = SOURCE.read_text()
    assert "io_uring" not in source
    assert "pread(" in source
    dockerfile = (SOURCE.parent / "Dockerfile").read_text()
    assert "COPY _ple_reader.c /opt/vonk/_ple_reader.c" in dockerfile
    assert "p.write_bytes(Path('/opt/vonk/_ple_reader.c').read_bytes())" in dockerfile
