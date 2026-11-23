"""The upstream providers a kit pin reads."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from recipe_tooling import catalog_hf_model
from recipe_tooling.kit_pins.model import (
    ABBREVIATED_SHA,
    IMAGE_DIGEST,
    MAX_METADATA_LAYER,
    SHA,
    KitError,
)
from recipe_tooling.paths import REPO_ROOT
from recipe_tooling.upstream_http import UpstreamBusy
from recipe_tooling.upstream_http import fetch as fetch_url

ROOT = REPO_ROOT

# ---------------------------------------------------------------------------
# Upstream access
# ---------------------------------------------------------------------------


class GitHub:
    """Immutable-commit reads through a GitHub API callable (``path -> JSON``)."""

    def __init__(self, api: Callable[[str], Any]) -> None:
        self.api = api

    def read(self, repo: str, commit: str, path: str) -> str:
        try:
            document = self.api(f"repos/{repo}/contents/{path}?ref={commit}")
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError) as error:
            raise KitError(
                f"cannot read {repo}@{commit[:12]}:{path}: {error}"
            ) from error
        if document.get("encoding") != "base64":
            raise KitError(f"{repo}@{commit[:12]}:{path} is not a readable file")
        return base64.b64decode(document["content"]).decode("utf-8", "replace")

    def resolve(self, repo: str, ref: str) -> tuple[str, str]:
        """``(commit, date)`` of a tag, branch or commit (annotated tags dereferenced)."""
        try:
            document = self.api(f"repos/{repo}/commits/{ref}")
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError) as error:
            raise KitError(f"cannot resolve {repo}@{ref}: {error}") from error
        commit = document["sha"]
        if not SHA.match(commit):
            raise KitError(f"{repo}@{ref}: provider returned a non-Git revision")
        return commit, str(document["commit"]["committer"]["date"])[:10]

    def kind(self, repo: str, ref: str) -> str:
        """What ``ref`` names in ``repo``: ``commit`` (a hex id), ``branch`` or ``tag``.

        A branch is told by the repository having a head of that name; anything else the
        provider resolves is a tag. (A name that is both is a branch, as ``git checkout``
        would treat it.)
        """
        if ABBREVIATED_SHA.match(ref):
            return "commit"
        try:
            self.api(f"repos/{repo}/git/ref/heads/{ref}")
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError):
            return "tag"
        return "branch"


class HuggingFace:
    """Anonymous reads of Hugging Face model repositories through ``path -> JSON``.

    Paths are relative to ``https://huggingface.co/api/``. No token is ever sent: a
    repository that is not anonymously readable cannot be locked.
    """

    def __init__(
        self,
        api: Callable[[str], Any],
        inventory: Callable[[str, str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.api = api
        self._inventory = inventory

    def inventory(self, repo: str, revision: str) -> list[dict[str, Any]]:
        """The files of a snapshot (``tools/catalog-hf-model``'s artifact records)."""
        if self._inventory is None:
            raise KitError("no Hugging Face file listing is configured")
        return self._inventory(repo, revision)

    def resolve(self, repo: str, ref: str) -> tuple[str, str]:
        """``(revision, date)`` of a commit, branch or tag of a model repository."""
        try:
            document = self.api(
                f"models/{repo}/revision/{urllib.parse.quote(ref, safe='')}"
            )
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError) as error:
            raise KitError(f"cannot resolve {repo}@{ref}: {error}") from error
        revision = document.get("sha")
        if not isinstance(revision, str) or not SHA.match(revision):
            raise KitError(f"{repo}@{ref}: provider returned a non-Git revision")
        return revision, str(document.get("lastModified") or "")[:10]

    def kind(self, repo: str, ref: str) -> str:
        if ABBREVIATED_SHA.match(ref):
            return "commit"
        try:
            refs = self.api(f"models/{repo}/refs")
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError):
            return "tag"
        branches = {b.get("name") for b in refs.get("branches", [])}
        return "branch" if ref in branches else "tag"

    def head(self, repo: str) -> str | None:
        try:
            document = self.api(f"models/{repo}")
        except UpstreamBusy:
            raise
        except (RuntimeError, KeyError):
            return None
        sha = document.get("sha")
        return sha if isinstance(sha, str) and SHA.match(sha) else None


class Registry:
    """Image tag to digest through an OCI registry's anonymous pull access.

    ``fetch(url, headers) -> (status, response headers, body)``; response header names
    are lower case. The answer for a tag (or a digest the kit names) is the digest of the
    ``linux/arm64`` manifest: a multi-platform index is followed to its arm64 entry, a
    single-platform manifest is its own answer. That is the digest the Dockerfile's ``FROM``
    carries, because the platform builds and runs on arm64 only.
    """

    ACCEPT = (
        "application/vnd.oci.image.index.v1+json, "
        "application/vnd.docker.distribution.manifest.list.v2+json, "
        "application/vnd.oci.image.manifest.v1+json, "
        "application/vnd.docker.distribution.manifest.v2+json"
    )

    def __init__(
        self, fetch: Callable[[str, dict[str, str]], tuple[int, dict[str, str], bytes]]
    ) -> None:
        self.fetch = fetch
        # Answers within one run: many recipes run the same image, and anonymous pulls
        # are rate limited (a tag is read once per run; digests never change).
        self.answers: dict[tuple[str, str, str], tuple[int, dict[str, str], bytes]] = {}

    @staticmethod
    def split(repo: str) -> tuple[str, str]:
        """``(registry host, repository path)`` of an image repository name."""
        first, _, rest = repo.partition("/")
        host = "registry-1.docker.io"
        if rest and first in {"docker.io", "index.docker.io", host}:
            repo = rest  # Docker Hub spelled with its registry host
        elif rest and ("." in first or ":" in first or first == "localhost"):
            return first, rest
        return host, repo if "/" in repo else f"library/{repo}"

    def _get(self, repo: str, kind: str, ref: str) -> tuple[int, dict[str, str], bytes]:
        """``GET /v2/<path>/<kind>/<ref>`` with an anonymous pull token when asked for."""
        key = (repo.lower(), kind, ref)
        if key in self.answers:
            return self.answers[key]
        host, path = self.split(repo)
        url = f"https://{host}/v2/{path}/{kind}/{urllib.parse.quote(ref, safe=':')}"
        headers = {"Accept": self.ACCEPT}
        status, answer, body = self.fetch(url, headers)
        if status == 401:
            token = self._token(answer.get("www-authenticate", ""), path)
            status, answer, body = self.fetch(url, {**headers, "Authorization": token})
        if status == 200:
            self.answers[key] = (status, answer, body)
        return status, answer, body

    def resolve(self, repo: str, ref: str) -> str:
        """The ``linux/arm64`` manifest digest of ``repo:ref`` (or ``repo@digest``)."""
        status, answer, body = self._get(repo, "manifests", ref)
        if status != 200:
            raise KitError(f"cannot resolve {repo}:{ref}: HTTP {status}")
        try:
            document = json.loads(body)
        except ValueError as error:
            raise KitError(f"{repo}:{ref}: the registry sent no manifest") from error
        if "manifests" not in document:
            digest = answer.get("docker-content-digest") or (
                "sha256:" + hashlib.sha256(body).hexdigest()
            )
            if not IMAGE_DIGEST.match(digest):
                raise KitError(f"{repo}:{ref}: the registry sent no digest")
            return digest
        arm64 = [
            m["digest"]
            for m in document["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m.get("platform", {}).get("architecture") == "arm64"
        ]
        if len(arm64) != 1 or not IMAGE_DIGEST.match(arm64[0]):
            raise KitError(f"{repo}:{ref} has no single linux/arm64 manifest")
        return arm64[0]

    def _document(self, repo: str, kind: str, ref: str) -> bytes:
        status, _, body = self._get(repo, kind, ref)
        if status != 200:
            raise KitError(f"cannot read {repo} {kind} {ref}: HTTP {status}")
        return body

    def provenance(self, repo: str, digest: str, source: str) -> tuple[str, str] | None:
        """``(commit, build date)`` of the GitHub ``source`` that ``repo@digest`` records.

        An image states what it was built from in its OCI labels
        (``org.opencontainers.image.revision``, its ``.source`` naming ``source`` when
        set) or, as eugr's images do, in the ``build-metadata.yaml`` its last layer adds
        (``build_script_commit``, ``build_date``). ``None`` when it records neither: a
        build commit is read, never guessed.
        """
        manifest = json.loads(self._document(repo, "manifests", digest))
        config = json.loads(self._document(repo, "blobs", manifest["config"]["digest"]))
        labels = (config.get("config") or {}).get("Labels") or {}
        revision = str(labels.get("org.opencontainers.image.revision", ""))
        named = str(labels.get("org.opencontainers.image.source", ""))
        if SHA.match(revision) and named.removesuffix(".git").rstrip("/").lower() in {
            "",
            f"https://github.com/{source}".lower(),
        }:
            created = labels.get("org.opencontainers.image.created") or config.get(
                "created", ""
            )
            return revision, str(created)[:10]
        text = self.build_metadata(repo, digest)
        if text is None:
            return None
        commit = re.search(
            r"^build_script_commit:\s*([0-9a-f]{40})\s*$", text, re.MULTILINE
        )
        if commit is None:
            return None
        built = re.search(r"^build_date:\s*(\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
        return commit.group(1), "" if built is None else built.group(1)

    def build_metadata(self, repo: str, digest: str) -> str | None:
        """The ``build-metadata.yaml`` the image's last layer adds, as eugr's images do.

        ``None`` when the last layer is not a small metadata layer or carries no such file.
        """
        manifest = json.loads(self._document(repo, "manifests", digest))
        layers = manifest.get("layers") or []
        if not layers or int(layers[-1].get("size", 0)) > MAX_METADATA_LAYER:
            return None
        try:
            data = self._document(repo, "blobs", layers[-1]["digest"])
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                member = next(
                    (
                        m
                        for m in archive.getmembers()
                        if m.isfile() and m.name.endswith("build-metadata.yaml")
                    ),
                    None,
                )
                handle = None if member is None else archive.extractfile(member)
                return None if handle is None else handle.read().decode()
        except (tarfile.TarError, OSError, UnicodeDecodeError):
            return None

    def _token(self, challenge: str, path: str) -> str:
        fields = dict(re.findall(r'(\w+)="([^"]*)"', challenge))
        realm = fields.get("realm")
        if not challenge.lower().startswith("bearer") or not realm:
            raise KitError(f"registry for {path} wants credentials; none are used")
        query = urllib.parse.urlencode(
            {
                "service": fields.get("service", ""),
                "scope": fields.get("scope") or f"repository:{path}:pull",
            }
        )
        status, _, body = self.fetch(f"{realm}?{query}", {})
        try:
            document = json.loads(body)
            token = document.get("token") or document["access_token"]
        except (ValueError, KeyError, AttributeError) as error:
            raise KitError(
                f"registry for {path} refused an anonymous token (HTTP {status})"
            ) from error
        return f"Bearer {token}"


@dataclass(frozen=True, slots=True)
class Upstream:
    """Where each datasource is resolved: GitHub, Hugging Face and OCI registries."""

    github: GitHub
    hub: HuggingFace | None = None
    registry: Registry | None = None
    images: Callable[[str, str, str], list[dict[str, Any]]] | None = (
        None  # a kit's images
    )

    def hf(self) -> HuggingFace:
        if self.hub is None:
            raise KitError("no Hugging Face access is configured")
        return self.hub

    def oci(self) -> Registry:
        if self.registry is None:
            raise KitError("no registry access is configured")
        return self.registry


def fetch(url: str, headers: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "vonk-forge-recipes", **headers}
    )
    try:
        return fetch_url(
            request,
            lambda response: (
                response.status,
                {k.lower(): v for k, v in response.headers.items()},
                response.read(50_000_000),
            ),
        )
    except urllib.error.HTTPError as error:
        return (
            error.code,
            {k.lower(): v for k, v in error.headers.items()},
            error.read(1_000_000),
        )
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach {url}: {error.reason}") from error


def _hub_api(path: str) -> Any:
    status, _, body = fetch(f"https://huggingface.co/api/{path}", {})
    if status != 200:
        raise RuntimeError(f"HTTP {status} for {path}")
    return json.loads(body)


def _hub_inventory(repo: str, revision: str) -> list[dict[str, Any]]:
    """The files of a snapshot, listed and digested as ``tools/catalog-hf-model`` does."""
    base = f"https://huggingface.co/api/models/{urllib.parse.quote(repo, safe='/')}"
    tree = catalog_hf_model._get_tree(
        f"{base}/tree/{revision}?recursive=true&expand=true&limit=100"
    )
    return catalog_hf_model.inventory(repo, revision, tree)
