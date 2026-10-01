"""Renew repository-scoped GitHub App credentials at each git/gh boundary."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any


class RefreshAppAuth:
    def __init__(self, client_id: str, private_key: str, repository: str) -> None:
        if not client_id or not private_key or len(repository.split("/")) != 2:
            raise RuntimeError(
                "Refresh App is not configured: set REFRESH_APP_CLIENT_ID, "
                "REFRESH_APP_PRIVATE_KEY and GITHUB_REPOSITORY; install it only "
                "on the recipe repository."
            )
        self.client_id = client_id
        self.private_key = private_key
        self.repository = repository
        self._token = ""
        self._expires = 0.0

    def _request(
        self, method: str, path: str, token: str, body: dict[str, Any] | None = None
    ) -> Any:
        request = urllib.request.Request(
            f"https://api.github.com{path}",
            data=None if body is None else json.dumps(body).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "User-Agent": "vonk-recipe-refresh",
                "X-GitHub-Api-Version": "2026-03-10",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Refresh App {method} {path} refused (HTTP {error.code}); "
                "check its installation and repository permissions."
            ) from None
        except urllib.error.URLError:
            raise RuntimeError(
                "Refresh App API is unreachable; no credential fallback."
            ) from None
        return json.loads(data) if data else None

    def _jwt(self) -> str:
        def encoded(value: bytes) -> str:
            return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

        now = int(time.time())
        header = encoded(b'{"alg":"RS256","typ":"JWT"}')
        claims = encoded(
            json.dumps(
                {"iat": now - 60, "exp": now + 600, "iss": self.client_id}
            ).encode()
        )
        payload = f"{header}.{claims}"
        # NamedTemporaryFile is owner-only; the key is deleted before any gh/git child.
        with tempfile.NamedTemporaryFile() as key:
            key.write(self.private_key.encode())
            key.flush()
            result = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", key.name],
                input=payload.encode(),
                capture_output=True,
                timeout=15,
                check=False,
            )
        if result.returncode:
            raise RuntimeError("Refresh App private key could not sign a JWT.")
        return f"{payload}.{encoded(result.stdout)}"

    def environment(self) -> dict[str, str]:
        if time.time() >= self._expires - 300:
            jwt = self._jwt()
            installation = self._request(
                "GET", f"/repos/{self.repository}/installation", jwt
            )
            access = self._request(
                "POST",
                f"/app/installations/{installation['id']}/access_tokens",
                jwt,
                {
                    "repositories": [self.repository.split("/")[1]],
                    "permissions": {
                        "contents": "write",
                        "pull_requests": "write",
                        "issues": "write",
                        "actions": "write",
                    },
                },
            )
            expires = datetime.fromisoformat(access["expires_at"]).timestamp()
            if expires <= time.time() + 300 or not access["token"]:
                raise RuntimeError(
                    "Refresh App returned an unusable installation token."
                )
            self._token, self._expires = access["token"], expires
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::add-mask::{self._token}", flush=True)
        # Clear other Git credential helpers, then use gh with this renewed token.
        return {
            "GH_TOKEN": self._token,
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "credential.https://github.com.helper",
            "GIT_CONFIG_VALUE_0": "",
            "GIT_CONFIG_KEY_1": "credential.https://github.com.helper",
            "GIT_CONFIG_VALUE_1": "!gh auth git-credential",
        }

    def close(self) -> None:
        if self._token and time.time() < self._expires:
            self._request("DELETE", "/installation/token", self._token)
        self._token = ""
        self.private_key = ""
