from __future__ import annotations

import runpy

import pytest


TOOL = runpy.run_path("tools/build-catalog-index")


def test_multistage_digest_pins_are_manifested_once_and_stage_reuse_is_internal() -> None:
    content = b"""FROM --platform=linux/arm64 registry.example/base:builder@sha256:%s AS build
RUN true
FROM registry.example/base:runtime@sha256:%s
COPY --from=build /x /x
FROM registry.example/base:runtime@sha256:%s
""" % (b"a" * 64, b"b" * 64, b"b" * 64)
    assert TOOL["_dockerfile_build_inputs"](content, label="Dockerfile") == [
        {"kind": "oci-image", "reference": "registry.example/base:builder@sha256:" + "a" * 64, "platform": "linux/arm64"},
        {"kind": "oci-image", "reference": "registry.example/base:runtime@sha256:" + "b" * 64, "platform": "linux/arm64"},
    ]


@pytest.mark.parametrize(
    "content, message",
    [
        (b"FROM registry.example/base:latest\n", "digest pinned"),
        (b"FROM ${BASE}\n", "digest pinned"),
        (b"FROM --platform=linux/amd64 registry.example/base@sha256:" + b"a" * 64 + b"\n", "ARM64"),
        (b"FROM later\nFROM registry.example/base@sha256:" + b"a" * 64 + b" AS later\n", "digest pinned"),
    ],
)
def test_build_inputs_reject_unsafe_external_from(content: bytes, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        TOOL["_dockerfile_build_inputs"](content, label="Dockerfile")
