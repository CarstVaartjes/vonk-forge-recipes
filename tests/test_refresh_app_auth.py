"""Reject stale/bot credentials and retain App authentication across long refreshes."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "refresh_app_auth", ROOT / "tools/_refresh_auth.py"
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class AppAuthTests(unittest.TestCase):
    def test_missing_configuration_refuses_instead_of_using_actions_token(self) -> None:
        with (
            patch.dict(os.environ, {"GH_TOKEN": "actions-token"}),
            self.assertRaisesRegex(RuntimeError, "not configured"),
        ):
            module.RefreshAppAuth("", "", "CarstVaartjes/vonk-forge-recipes")

    def test_long_run_renews_and_git_uses_the_same_repository_scoped_token(
        self,
    ) -> None:
        auth = module.RefreshAppAuth(
            "client", "key", "CarstVaartjes/vonk-forge-recipes"
        )
        clock = [1000.0]
        requests = []

        def request(method, path, token, body=None):
            requests.append((method, path, body))
            if method == "GET":
                return {"id": 42}
            if method == "POST":
                return {
                    "token": f"installation-{len(requests)}",
                    "expires_at": datetime.fromtimestamp(
                        clock[0] + 3600, UTC
                    ).isoformat(),
                }
            return None

        with (
            patch.object(auth, "_jwt", return_value="jwt"),
            patch.object(auth, "_request", side_effect=request),
            patch.object(module.time, "time", side_effect=lambda: clock[0]),
        ):
            first = auth.environment()
            clock[0] += 3200
            self.assertEqual(first, auth.environment())
            clock[0] += 200
            second = auth.environment()
            self.assertNotEqual(first["GH_TOKEN"], second["GH_TOKEN"])
            self.assertEqual(second["GIT_CONFIG_VALUE_0"], "")
            self.assertEqual(second["GIT_CONFIG_VALUE_1"], "!gh auth git-credential")
            auth.close()
        posts = [r for r in requests if r[0] == "POST"]
        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0][2]["repositories"], ["vonk-forge-recipes"])
        self.assertNotIn("administration", posts[0][2]["permissions"])
        self.assertEqual(requests[-1][:2], ("DELETE", "/installation/token"))
        self.assertEqual(auth.private_key, "")

    def test_invalid_key_never_leaks_openssl_diagnostics(self) -> None:
        auth = module.RefreshAppAuth("client", "THIS IS SECRET", "owner/repo")
        with self.assertRaisesRegex(RuntimeError, "could not sign") as caught:
            auth._jwt()
        self.assertNotIn("THIS IS SECRET", str(caught.exception))

    def test_jwt_is_really_rsa_signed_and_key_file_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            private = Path(temp) / "key.pem"
            public = Path(temp) / "public.pem"
            subprocess.run(
                ["openssl", "genrsa", "-out", str(private), "2048"],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["openssl", "rsa", "-in", str(private), "-pubout", "-out", str(public)],
                check=True,
                capture_output=True,
            )
            auth = module.RefreshAppAuth("client", private.read_text(), "owner/repo")
            signed_paths = []
            original_run = subprocess.run

            def tracked_run(command, **kwargs):
                signed_paths.append(Path(command[-1]))
                self.assertEqual(signed_paths[-1].stat().st_mode & 0o777, 0o600)
                return original_run(command, **kwargs)

            with patch.object(module.subprocess, "run", side_effect=tracked_run):
                token = auth._jwt()
            header, claims, signature = token.split(".")
            decoded = json.loads(
                base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4))
            )
            self.assertEqual(decoded["iss"], "client")
            self.assertLessEqual(decoded["exp"] - decoded["iat"], 660)
            sig = Path(temp) / "signature"
            sig.write_bytes(
                base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4))
            )
            verified = original_run(
                [
                    "openssl",
                    "dgst",
                    "-sha256",
                    "-verify",
                    str(public),
                    "-signature",
                    str(sig),
                ],
                input=f"{header}.{claims}".encode(),
                capture_output=True,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertFalse(signed_paths[0].exists())


class RefusedAppAuthTests(unittest.TestCase):
    def test_denied_renewal_never_returns_old_or_personal_credentials(self) -> None:
        auth = module.RefreshAppAuth("client", "key", "owner/repo")
        auth._token = "expired-token"
        auth._expires = 0
        with (
            patch.dict(os.environ, {"GH_TOKEN": "personal-token"}),
            patch.object(auth, "_jwt", return_value="jwt"),
            patch.object(auth, "_request", side_effect=RuntimeError("HTTP 403")),
            self.assertRaisesRegex(RuntimeError, "HTTP 403"),
        ):
            auth.environment()
