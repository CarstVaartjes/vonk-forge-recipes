"""A thin, safe wrapper around the ``vonkctl`` command line in JSON mode.

Every call goes through :class:`Vonkctl`, so the owner's profiles can never be
written by accident: a profile write must name its profile explicitly, and
profiles 1-3 are refused unless the caller opts in for one deliberate restore.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

OWNER_PROFILES = frozenset({1, 2, 3})
PROFILE_WRITES = frozenset({"add", "remove", "configure", "import", "load", "cancel"})
_KEY_NAMESPACE = uuid.UUID("6f1d1a5e-5b0c-4d7e-9a53-0d6c5b2a7e11")

# (argv, timeout seconds) -> (exit code, stdout, stderr)
Runner = Callable[[Sequence[str], float], tuple[int, str, str]]


def request_key(*parts: object) -> str:
    """A deterministic request UUID, so a restarted run reconnects to its work."""
    return str(uuid.uuid5(_KEY_NAMESPACE, "|".join(str(part) for part in parts)))


class OwnerProfileError(ValueError):
    """A write that would touch a profile the sweep does not own."""


class VonkctlError(RuntimeError):
    """vonkctl failed or returned an error document."""

    def __init__(self, argv: Sequence[str], reply: Reply | None, message: str) -> None:
        super().__init__(message)
        self.argv = tuple(argv)
        self.reply = reply

    @property
    def code(self) -> str:
        document = self.reply.document if self.reply else None
        if isinstance(document, dict):
            return str(document.get("code") or document.get("error_type") or "")
        return ""

    def infrastructure(self, *, reviewing: bool = False) -> bool:
        return is_infrastructure(self.reply, reviewing=reviewing)


RECIPE_ERROR_CODES = frozenset(
    {
        "recipe.invalid_definition",
        "recipe.invalid_document",
        "recipe.invalid_option",
        "dockerfile.heredoc_forbidden",
        "recipe.digest_mismatch",
    }
)


def is_infrastructure(reply: Reply | None, *, reviewing: bool = False) -> bool:
    """Only an explicit recipe-data code attributes a CLI error to a recipe.

    HTTP failures, unknown outcomes and Controller codes remain infrastructure,
    regardless of status or free-text detail. Re-read and retry without blame.
    """
    document = reply.document if reply is not None else None
    if not isinstance(document, dict):
        return True
    code = str(document.get("code") or "")
    if code.startswith(("http.", "controller.")):
        return True
    if any(
        document.get(field) is not None
        for field in ("status", "status_code", "http_status")
    ):
        return True
    return code not in RECIPE_ERROR_CODES


class VonkctlTimeout(VonkctlError):
    """The command did not finish within its time limit."""


@dataclass(frozen=True)
class Reply:
    argv: tuple[str, ...]
    exit_code: int
    document: Any
    stderr: str

    @property
    def is_error_document(self) -> bool:
        return isinstance(self.document, dict) and "error_type" in self.document

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.is_error_document

    @property
    def is_not_found(self) -> bool:
        """The installed CLI's declared Controller absence, never a read error guess."""
        document = self.document
        return (
            self.exit_code != 0
            and isinstance(document, dict)
            and document.get("error_type") == "control_api"
            and document.get("code") == "controller.not_found"
        )

    @property
    def error_text(self) -> str:
        if isinstance(self.document, dict):
            parts = [
                str(self.document.get(name))
                for name in ("code", "detail", "error")
                if self.document.get(name)
            ]
            if parts:
                return ": ".join(parts)
        return (self.stderr or f"exit {self.exit_code}").strip()[:500]


def _kill_group(process: subprocess.Popen[str]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()


def subprocess_runner(argv: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """Run one command in its own session, so an interrupt or a timeout really ends it.

    The child is not in our terminal's process group (Ctrl-C reaches only us) and it
    inherits no ignored SIGINT; on an interrupt, a timeout or any other exit from this
    function its whole process group is killed.
    """
    process = subprocess.Popen(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        _kill_group(process)
        process.communicate()
        raise VonkctlTimeout(
            argv, None, f"{argv[0]} timed out after {timeout}s"
        ) from error
    except BaseException:
        _kill_group(process)
        process.wait()
        raise
    return process.returncode, out, err


def _parse(text: str) -> Any:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        # Tolerate progress lines before the final document.
        for line in reversed(text.splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


class Vonkctl:
    def __init__(
        self,
        executable: str = "vonkctl",
        *,
        runner: Runner = subprocess_runner,
        timeout: float = 180.0,
    ) -> None:
        self.executable = executable
        self.runner = runner
        self.timeout = timeout

    def run(
        self,
        *args: str,
        profile: int | None = None,
        timeout: float | None = None,
        allow_owner_write: bool = False,
    ) -> Reply:
        self._guard(args, profile, allow_owner_write)
        argv = [self.executable, "--json", "--no-input"]
        if profile is not None:
            argv += ["--profile", str(profile)]
        argv += list(args)
        try:
            code, out, err = self.runner(argv, timeout or self.timeout)
        except VonkctlTimeout as error:
            return Reply(tuple(argv), 124, None, str(error))
        except OSError as error:
            # The binary is missing (``vonkctl update --apply`` replaces it atomically) or
            # cannot be spawned: infrastructure, like a timeout. No document, so the
            # callers pause and retry.
            return Reply(tuple(argv), 127, None, f"{type(error).__name__}: {error}")
        document = _parse(out)
        if document is None:
            document = _parse(err)
        return Reply(tuple(argv), code, document, err)

    def call(
        self,
        *args: str,
        profile: int | None = None,
        timeout: float | None = None,
        tolerate: Sequence[int] = (),
        allow_owner_write: bool = False,
    ) -> Any:
        """Run a command and return its document; raise on failure.

        ``tolerate`` lists exit codes that still carry a meaningful document
        (a blocked review exits 2, a partial operation exits 1).
        """
        reply = self.run(
            *args, profile=profile, timeout=timeout, allow_owner_write=allow_owner_write
        )
        failed = reply.is_error_document or reply.document is None
        if reply.exit_code != 0 and reply.exit_code not in tolerate:
            failed = True
        if failed:
            raise VonkctlError(reply.argv, reply, reply.error_text)
        return reply.document

    @staticmethod
    def _guard(
        args: Sequence[str], profile: int | None, allow_owner_write: bool
    ) -> None:
        if len(args) >= 2 and args[0] == "profile" and args[1] in PROFILE_WRITES:
            if profile is None:
                raise OwnerProfileError(
                    "profile writes need an explicit --profile (the default is the owner's profile 1)"
                )
            if profile in OWNER_PROFILES and not allow_owner_write:
                raise OwnerProfileError(f"refusing to write owner profile {profile}")
        if args and args[0] == "run":
            raise OwnerProfileError(
                "vonkctl run edits a profile; the sweep never uses it"
            )
