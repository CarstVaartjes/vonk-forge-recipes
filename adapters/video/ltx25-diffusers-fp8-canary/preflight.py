#!/usr/bin/env python3
"""Verify LTX-2.5 license acknowledgement and gated access without model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_REPOSITORY = "Lightricks/LTX-2.5-Diffusers"
MODEL_REVISION = "426936f8b22dc28e4def61e515478b0b7e4a53cc"
MODEL_URL = f"https://huggingface.co/{MODEL_REPOSITORY}"
LICENSE_URL = (
    "https://github.com/Lightricks/LTX-2/blob/"
    "a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2_x"
)
LICENSE_SHA256 = "be75acae5c99b0fb16ed6cfbf8f731e5121a729bef112d20337699407e796451"
PROBE_PATH = "audio_vae/config.json"
PROBE_BYTES = 505
PROBE_GIT_BLOB_SHA1 = "6759c3aa9ed4772ccd7c9c8dba1378f7b3ac7aba"
PROBE_URL = f"{MODEL_URL}/resolve/{MODEL_REVISION}/{PROBE_PATH}?download=true"
USER_AGENT = "vonk-forge-ltx25-access-preflight/1"


class PreflightError(RuntimeError):
    """An actionable preflight failure that never contains a credential."""


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Never forward the Hugging Face bearer token to another origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old = urllib.parse.urlsplit(req.full_url)
        new = urllib.parse.urlsplit(newurl)
        if (new.scheme, new.hostname, new.port) != (old.scheme, old.hostname, old.port):
            raise PreflightError(
                "Hugging Face redirected the access probe to another origin; "
                "the token was not forwarded and no model weights were downloaded."
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _token_from_file(path: Path) -> str:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PreflightError(f"cannot read token file {path}: {error.strerror}") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise PreflightError("token file must be a regular file and not a symbolic link")
    if metadata.st_uid != os.geteuid():
        raise PreflightError("token file must be owned by the user running the preflight")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise PreflightError("token file permissions must deny group and other access (mode 0600)")
    if metadata.st_size <= 0 or metadata.st_size > 4096:
        raise PreflightError("token file must contain one non-empty token and be at most 4096 bytes")
    try:
        token = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise PreflightError("token file must contain one ASCII Hugging Face token") from error
    if not token.startswith("hf_") or any(character.isspace() for character in token):
        raise PreflightError("token file must contain one Hugging Face token beginning with hf_")
    return token


def _verify_license_acknowledgement(value: str) -> None:
    if value != LICENSE_SHA256:
        raise PreflightError(
            "license acknowledgement does not match the pinned LTX-2 Community "
            f"License at {LICENSE_URL}; review it and pass SHA-256 {LICENSE_SHA256}"
        )


def _verify_gated_access(token: str, *, opener=None) -> str:
    request = urllib.request.Request(
        PROBE_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    opener = opener or urllib.request.build_opener(_SameOriginRedirectHandler())
    try:
        with opener.open(request, timeout=30) as response:
            final_url = urllib.parse.urlsplit(response.geturl())
            if final_url.scheme != "https" or final_url.hostname != "huggingface.co":
                raise PreflightError(
                    "access probe left huggingface.co; the response was rejected and no "
                    "model weights were downloaded"
                )
            payload = response.read(PROBE_BYTES + 1)
    except urllib.error.HTTPError as error:
        if error.code == 401:
            reason = (
                "Hugging Face rejected the token (401 GatedRepo). Confirm that the token "
                "is valid and belongs to the account approved at the official model page."
            )
        elif error.code == 403:
            reason = (
                "Hugging Face denied access (403). Review and accept the model gate and "
                "license with the same account that owns this token."
            )
        elif error.code == 404:
            reason = (
                "the pinned model revision or probe file is unavailable to this account "
                "(404); do not start the model download"
            )
        else:
            reason = f"Hugging Face returned HTTP {error.code}"
        raise PreflightError(
            f"{reason} Model: {MODEL_URL}. No model weights were downloaded."
        ) from error
    except urllib.error.URLError as error:
        raise PreflightError(
            f"could not reach Hugging Face ({error.reason}); no model weights were downloaded"
        ) from error

    if len(payload) != PROBE_BYTES:
        raise PreflightError(
            f"pinned access probe has {len(payload)} bytes, expected {PROBE_BYTES}; "
            "no model weights were downloaded"
        )
    git_blob = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()
    if git_blob != PROBE_GIT_BLOB_SHA1:
        raise PreflightError(
            "pinned access probe content changed; no model weights were downloaded"
        )
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acknowledge the exact pinned LTX-2 Community license and verify the "
            "same-account Hugging Face token using one 505-byte gated config file."
        )
    )
    parser.add_argument(
        "--token-file",
        required=True,
        type=Path,
        help="0600 file containing the approved account's Hugging Face read token",
    )
    parser.add_argument(
        "--accept-license-sha256",
        required=True,
        help="SHA-256 of the exact pinned LTX-2 Community license after reviewing it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        _verify_license_acknowledgement(args.accept_license_sha256)
        token = _token_from_file(args.token_file)
        probe_sha256 = _verify_gated_access(token)
    except PreflightError as error:
        print(f"LTX-2.5 preflight FAILED: {error}", file=sys.stderr)
        return 2

    print(
        json.dumps(
            {
                "status": "passed",
                "license": {
                    "acknowledged": True,
                    "sha256": LICENSE_SHA256,
                    "url": LICENSE_URL,
                },
                "model": {
                    "repository": MODEL_REPOSITORY,
                    "revision": MODEL_REVISION,
                    "probe_bytes": PROBE_BYTES,
                    "probe_path": PROBE_PATH,
                    "probe_sha256": probe_sha256,
                },
                "next": (
                    "Configure this same approved account token in every target Spark's "
                    "root-owned huggingface_curl_config before installation."
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
