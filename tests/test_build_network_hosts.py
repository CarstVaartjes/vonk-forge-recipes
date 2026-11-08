from __future__ import annotations

import glob
import json
import posixpath
import re
import shlex
import tempfile
import unittest
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlparse

from documents import recipe as document_recipe

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# The build's network reach is a static question: which commands in the Dockerfile, and in the
# context scripts its RUN lines execute, fetch from the network, and from which hosts. The
# analysis is a line-and-token reading, not a shell parser. Limits: exec-form RUN, `bash -c`,
# computed script paths and install arguments held in variables are not followed.
INSTALL = re.compile(r"(?:\bpip3?|-m pip|\buv pip)\s+install\b(?P<args>.*)")
PYPI = frozenset({"pypi.org", "files.pythonhosted.org"})
VALUE_FLAGS = {"-r", "-c", "-f", "--requirement", "--constraint", "--find-links"}
INDEX_FLAGS = {"-i", "--index-url", "--extra-index-url"}
OTHER_VALUE_FLAGS = {
    "--python",
    "--target",
    "-t",
    "--prefix",
    "--root",
    "--index-strategy",
}
SCRIPT_SUFFIXES = {".sh", ".bash", ".py"}
RUNNERS = {"bash", "sh", "source", "."}
DOCKERFILE_KEYWORDS = (
    r"FROM|ENV|LABEL|COPY|ADD|ARG|WORKDIR|USER|ENTRYPOINT|CMD|EXPOSE|VOLUME"
    r"|HEALTHCHECK|SHELL|STOPSIGNAL|ONBUILD"
)
OFFLINE_ENV = re.compile(
    r"\b(?:PIP_NO_INDEX|UV_OFFLINE)[= ]\s*(?:1|true|yes)\b", re.IGNORECASE
)
LOCAL_URL = re.compile(r"://(?:localhost|127\.0\.0\.1|\[::1\])(?:[:/]|$)")
PROBE = re.compile(
    r"^(?:python3?\b.*(?:\s-c\b|\s-\s*(?:<<|$)|\bimport\b)|command\s+-v\b|which\b|test\b|\["
    r"|.*\b(?:pip3?|uv pip)\s+(?:show|list|freeze)\b)"
)
LEADING_KEYWORDS = re.compile(r"^(?:(?:do|then|else|if|elif|while|until|!|time)\s+)+")
LEADING_ENV = re.compile(r"^(?:[A-Za-z_]\w*=\S*\s+)+")


class Fetch(NamedTuple):
    command: str
    hosts: frozenset[str]  # hosts the command needs; empty when not known


def commands(text: str) -> list[tuple[str, bool]]:
    """Shell commands of a text as (command, guarded).

    Comments, echo/printf/print lines and Dockerfile instructions other than RUN are dropped. A
    command is guarded when it only runs after a failed probe: `<probe> || cmd`, or inside an
    `if [!] <probe>` block. A probe is an import check, `command -v`, `which`, `test`, `[` or
    `pip show/list`.
    """
    text = re.sub(r"(?m)(?:^|\s)#.*$", "", text)
    text = re.sub(r"\\\n", " ", text)
    lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(rf"(?:{DOCKERFILE_KEYWORDS})\s", line, re.IGNORECASE):
            continue
        lines.append(re.sub(r"^RUN\s+(?:--\S+\s+)*", "", line))
    parts = re.split(r"(&&|\|\||;|\||\n)", "\n".join(lines))
    out: list[tuple[str, bool]] = []
    previous, separator, in_probe_if = "", "", False
    for index in range(0, len(parts), 2):
        raw = parts[index].strip()
        if index:
            separator = parts[index - 1]
        command = LEADING_KEYWORDS.sub("", raw)
        if not command:
            continue
        if command == "fi":
            in_probe_if = False
        elif re.match(r"if\s+(?:!\s*)?", raw) and PROBE.match(command):
            in_probe_if = True
        elif not re.match(r"(?:echo|printf)\s|print\(", command):
            guarded = in_probe_if or (separator == "||" and bool(PROBE.match(previous)))
            out.append((command, guarded))
        previous = command
    return out


def _words(command: str) -> list[str]:
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    while words and re.fullmatch(r"[A-Za-z_]\w*=.*", words[0]):
        words.pop(0)
    return words


def _host(url: str) -> str | None:
    return urlparse(url.removeprefix("git+")).hostname


def pip_fetch(command: str, offline_env: bool) -> Fetch | None:
    """The fetch of a pip/uv install, or None when it reaches no remote.

    An install fetches nothing with --no-index/--offline, or when --no-deps is set and every
    target is a local wheel, or a local source with --no-build-isolation (an isolated build
    downloads its build backend). Hosts: the default index is the PyPI pair; --index-url
    replaces it (PyPI keeps its file host); --extra-index-url and --find-links add theirs;
    a URL target adds its own, and with --no-deps and only URL targets nothing else.
    """
    match = INSTALL.search(command)
    if not match:
        return None
    where = match.group(0).split("&&")[0].strip()
    try:
        args = shlex.split(match.group("args"))
    except ValueError:
        return Fetch(where, frozenset())
    if offline_env or {"--no-index", "--offline"} & set(args):
        return None
    targets: list[str] = []
    index: str | None = None
    extra: set[str] = set()
    requirement_file = False
    flags = iter(args)
    for arg in flags:
        name, _, inline = arg.partition("=")
        if name in VALUE_FLAGS | INDEX_FLAGS | OTHER_VALUE_FLAGS:
            value = inline or next(flags, "")
            requirement_file = requirement_file or name in {"-r", "--requirement"}
            host = _host(value) if "://" in value else None
            if name in {"-i", "--index-url"}:
                index = host or "?"
            elif host and name in {"--extra-index-url", "-f", "--find-links"}:
                extra.add(host)
        elif not arg.startswith("-"):
            targets.append(arg)
    local = bool(targets) and all(t.startswith(("/", ".")) for t in targets)
    if (
        "--no-deps" in args
        and not requirement_file
        and local
        and ("--no-build-isolation" in args or all(t.endswith(".whl") for t in targets))
    ):
        return None
    urls = {h for t in targets if "://" in t and (h := _host(t))}
    url_only = (
        "--no-deps" in args
        and not requirement_file
        and bool(targets)
        and all("://" in t for t in targets)
    )
    hosts = set(extra) | urls
    if index and index != "?":
        hosts.add(index)
        if index == "pypi.org":
            hosts.add("files.pythonhosted.org")
    elif not index and not url_only:
        hosts |= PYPI
    return Fetch(where, frozenset(hosts))


def other_fetch(command: str) -> Fetch | None:
    words = _words(command)
    if words and words[0] == "sudo":
        words = words[1:]
    if not words or LOCAL_URL.search(command):
        return None
    tool, rest = words[0], words[1:]
    if tool in {"curl", "wget"}:
        fetching = True
    elif tool == "git":
        fetching = bool(rest) and (
            rest[0] in {"fetch", "pull"}
            or rest[:2] == ["submodule", "update"]
            or (rest[0] == "clone" and any("://" in w or "@" in w for w in rest))
        )
    elif tool in {"apt", "apt-get"}:
        verbs = [w for w in rest if not w.startswith("-")]
        local = verbs[1:] and all(w.endswith(".deb") for w in verbs[1:])
        fetching = (
            bool(verbs)
            and verbs[0] in {"install", "update", "upgrade", "download"}
            and not local
        )
    elif tool in {"npm", "pnpm", "yarn"}:
        fetching = bool(rest) and rest[0] in {"install", "add", "ci"}
    else:
        fetching = tool == "cargo" and bool(rest) and rest[0] in {"install", "fetch"}
    return Fetch(" ".join(words[:4]), frozenset()) if fetching else None


def fetches(text: str) -> list[Fetch]:
    """Network fetches in a Dockerfile or script text, probe-guarded installs excluded."""
    offline_env = bool(OFFLINE_ENV.search(text))
    found: list[Fetch] = []
    for command, guarded in commands(text):
        if guarded:
            continue
        fetch = pip_fetch(command, offline_env) or other_fetch(command)
        if fetch:
            found.append(fetch)
    return found


def _executed(text: str, cwd: str, script_dir: str | None = None) -> list[str]:
    """Absolute in-image paths a text executes: `bash x`, `python3 x`, `source x`, `. x`, `./x`.

    The directory follows `cd` and the starting directory; a leading `$VAR/` in a script is
    its own directory. chmod, test -x, sed and the like do not execute their argument.
    """
    paths: list[str] = []
    for command, _ in commands(text):
        words = _words(command)
        if not words:
            continue
        if words[0] == "cd" and len(words) > 1:
            cwd = posixpath.join(cwd, words[1])
            continue
        tool = words[0]
        if tool in RUNNERS or re.fullmatch(r"python[\d.]*", tool):
            if "-c" in words or "-m" in words:
                continue
            rest = [w for w in words[1:] if not w.startswith("-")]
            target = rest[0] if rest else ""
        else:
            target = tool if "/" in tool else ""
        if script_dir and target.startswith("$") and "/" in target:
            target = posixpath.join(script_dir, target.split("/", 1)[1])
        if Path(target).suffix in SCRIPT_SUFFIXES:
            paths.append(posixpath.normpath(posixpath.join(cwd, target)))
    return paths


def _copied(
    dockerfile_text: str, context: Path
) -> tuple[dict[str, Path], list[tuple[str, str]]]:
    """COPY/ADD results as {in-image path: context file}, and (WORKDIR, RUN text) per RUN."""
    joined = re.sub(r"\\\n", " ", re.sub(r"(?m)^\s*#.*$", "", dockerfile_text))
    workdir, copied, runs = "/", {}, []
    for line in joined.splitlines():
        keyword, _, rest = line.strip().partition(" ")
        keyword = keyword.upper()
        if keyword == "WORKDIR":
            workdir = posixpath.join(workdir, rest.strip())
        elif keyword == "RUN":
            runs.append((workdir, rest))
        elif keyword in {"COPY", "ADD"} and "--from=" not in rest:
            words = [w for w in rest.split() if not w.startswith("--")]
            if len(words) < 2 or words[0].startswith("http"):
                continue
            *sources, raw = words
            into_dir = raw.endswith("/") or raw in {".", ".."} or len(sources) > 1
            dest = posixpath.join(workdir, raw)
            for source in sources:
                for match in sorted(glob.glob(str(context / source))):
                    base = Path(match)
                    files = (
                        [f for f in sorted(base.rglob("*")) if f.is_file()]
                        if base.is_dir()
                        else [base]
                    )
                    for file in files:
                        if base.is_dir():
                            image = posixpath.join(
                                dest, file.relative_to(base).as_posix()
                            )
                        else:
                            image = (
                                posixpath.join(dest, file.name) if into_dir else dest
                            )
                        copied[posixpath.normpath(image)] = file
    return copied, runs


def run_scripts(dockerfile_text: str, context: Path) -> dict[str, str]:
    """Context scripts the build executes, as {context-relative path: text}.

    A script runs when a RUN line executes it, or a run script executes it by name (the
    transitive closure). A `$var` or `*` path segment matches any single segment, so
    `for m in ...; do bash .../mods/$m/run.sh; done` covers every mods/*/run.sh.
    """
    copied, runs = _copied(dockerfile_text, context)

    def matches(pattern: str) -> set[str]:
        regex = re.escape(pattern)
        regex = re.sub(r"\\\$\\\{\w+\\\}|\\\$\w+|\\\*", "[^/]+", regex)
        return {i for i in copied if re.fullmatch(regex, i)}

    executed: set[str] = set()
    for workdir, body in runs:
        for pattern in _executed(body, workdir):
            executed |= matches(pattern)
    pending = list(executed)
    while pending:
        image = pending.pop()
        file = copied[image]
        if file.suffix not in SCRIPT_SUFFIXES:
            continue
        directory = posixpath.dirname(image)
        text = file.read_text(encoding="utf-8", errors="replace")
        for pattern in _executed(text, directory, directory):
            for found in matches(pattern) - executed:
                executed.add(found)
                pending.append(found)
    return {
        copied[i].relative_to(context).as_posix(): copied[i].read_text(
            encoding="utf-8", errors="replace"
        )
        for i in sorted(executed)
        if copied[i].suffix in SCRIPT_SUFFIXES
    }


def build_fetches(dockerfile: Path, context: Path) -> list[Fetch]:
    """Fetches in the Dockerfile and in the context scripts its RUN lines execute."""
    text = dockerfile.read_text(encoding="utf-8")
    found = fetches(text)
    for name, script in run_scripts(text, context).items():
        found += [f._replace(command=f"{name}: {f.command}") for f in fetches(script)]
    return found


class BuildNetworkHostMetadataTests(unittest.TestCase):
    def test_declared_build_hosts_are_used_by_source_builds(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            network = recipe["execution"]["build"]["network"]
            self.assertEqual(
                len(network["hosts"]), len(set(network["hosts"])), path.name
            )
            dockerfile = ROOT / recipe["execution"]["build"]["dockerfile"]
            self.assertTrue(dockerfile.is_file(), path.name)

    def test_build_fetches_are_covered_by_declared_hosts(self) -> None:
        """The Spark builder gives an empty host list --network=none; a declared list must also
        hold every host the build's fetches need (Dockerfile and executed context scripts)."""
        problems: dict[str, list[str]] = {}
        for path in sorted((ROOT / "recipes").glob("*.json")):
            build = document_recipe(path.stem).execution.build
            declared = set(build.network.hosts)
            for fetch in build_fetches(
                ROOT / build.dockerfile, ROOT / build.context.path
            ):
                if not declared or not fetch.hosts <= declared:
                    problems.setdefault(path.stem, []).append(
                        f"{fetch.command} needs {sorted(fetch.hosts - declared)}"
                    )
        self.assertEqual({}, problems)

    def test_fetching_install_detection(self) -> None:
        def hosts(text: str) -> list[frozenset[str]]:
            return [f.hosts for f in fetches(text)]

        self.assertEqual([PYPI], hosts("RUN pip install foo"))
        self.assertEqual([PYPI], hosts("RUN python -m pip install -r /tmp/a.txt"))
        self.assertFalse(fetches("RUN pip install --no-index /tmp/a.whl"))
        self.assertFalse(fetches("RUN uv pip install --offline foo"))
        self.assertFalse(fetches("ENV PIP_NO_INDEX=1\nRUN pip install foo"))
        self.assertFalse(
            fetches(
                "RUN python -m pip install --no-deps --no-build-isolation /s && pip show x"
            )
        )
        self.assertFalse(fetches("RUN pip install --no-deps /tmp/a.whl"))
        self.assertFalse(fetches("RUN apt-get install -y ./a.deb"))
        self.assertFalse(fetches("RUN curl -fsS http://localhost:8000/health"))
        # an isolated source build downloads its build backend
        self.assertTrue(fetches("RUN pip install --no-deps /tmp/src"))
        # the command ends at the line: a later --no-index or apostrophe is not ours
        self.assertTrue(fetches("RUN pip install foo\nRUN echo --no-index"))
        self.assertTrue(fetches("RUN pip install foo\n# it's a comment"))
        for other in (
            "curl -O https://x/y",
            "wget https://x/y",
            "git clone https://x/y",
            "apt-get update",
        ):
            self.assertTrue(fetches(f"RUN {other}"), other)

    def test_comments_and_printed_text_are_not_fetches(self) -> None:
        self.assertFalse(fetches("RUN echo 'pip install foo' && true"))
        self.assertFalse(fetches("# pip install foo\nRUN true # curl https://x"))
        self.assertFalse(fetches('print("run pip install foo first")'))

    def test_probe_guarded_install_is_not_a_fetch(self) -> None:
        self.assertFalse(fetches("python3 -c 'import av' || uv pip install av"))
        self.assertFalse(fetches("python3 - <<'PY' || pip install av\nimport av\nPY"))
        self.assertFalse(fetches("command -v git || apt-get install -y git"))
        self.assertFalse(
            fetches("if ! python3 -c 'import av'; then pip install av; fi")
        )
        # a failure of anything else does not guard it
        self.assertTrue(fetches("make build || pip install av"))
        self.assertTrue(fetches("if true; then pip install av; fi"))

    def test_hosts_follow_the_index(self) -> None:
        def hosts(text: str) -> frozenset[str]:
            return fetches(text)[0].hosts

        torch = "download.pytorch.org"
        self.assertEqual(
            frozenset({torch}),
            hosts(f"pip install --index-url https://{torch}/whl/cu130 torch"),
        )
        self.assertEqual(
            PYPI | {torch},
            hosts(f"pip install --extra-index-url https://{torch}/whl/cu130 torch"),
        )
        self.assertEqual(
            PYPI, hosts("pip install --index-url=https://pypi.org/simple foo")
        )
        self.assertEqual(
            frozenset({"github.com"}),
            hosts("pip install --no-deps git+https://github.com/a/b@c"),
        )
        self.assertEqual(
            PYPI | {"github.com"}, hosts("pip install git+https://github.com/a/b@c")
        )

    def test_script_fetch_detection(self) -> None:
        dockerfiles = {
            "for": 'COPY mods /opt/m\nRUN for m in a; do bash "/opt/m/$m/run.sh"; done',
            "cd": "COPY mods /opt/m\nRUN set -e; cd /opt/m/a; FOO=1 bash run.sh",
            "workdir": "COPY mods/a/run.sh .\nWORKDIR /opt/w\nCOPY mods/a/run.sh .\nRUN ./run.sh",
            "bare": "WORKDIR /opt/m/a\nCOPY mods/a /opt/m/a\nRUN bash run.sh",
            "source": "COPY mods /opt/m\nRUN . /opt/m/a/run.sh",
        }
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            module = context / "mods" / "a"
            module.mkdir(parents=True)
            for name, body in {
                "run.sh": "#!/bin/bash\nuv pip install --quiet soundfile\n",
                "unused.sh": "pip install nothing\n",
            }.items():
                (module / name).write_text(body, encoding="utf-8")
            for name, dockerfile in dockerfiles.items():
                scripts = run_scripts(dockerfile, context)
                self.assertIn("mods/a/run.sh", scripts, name)
                self.assertNotIn("mods/a/unused.sh", scripts, name)
            (context / "Dockerfile").write_text(dockerfiles["for"], encoding="utf-8")
            found = build_fetches(context / "Dockerfile", context)
            self.assertEqual(
                ["mods/a/run.sh: uv pip install --quiet soundfile"],
                [f.command for f in found],
            )
            # executed by name from a run script, but not merely mentioned or chmod'ed
            (module / "run.sh").write_text(
                "chmod +x helper.sh\ncat unused.sh\n$DIR/helper.sh\n# bash unused.sh\n",
                encoding="utf-8",
            )
            (module / "helper.sh").write_text(
                "curl -fsSL https://x/y | sh\n", encoding="utf-8"
            )
            scripts = run_scripts(dockerfiles["for"], context)
            self.assertEqual({"mods/a/run.sh", "mods/a/helper.sh"}, set(scripts))
            # a probe-guarded install and a commented one are clean
            (module / "helper.sh").write_text(
                "python3 -c 'import av' || uv pip install av\n# pip install x\n",
                encoding="utf-8",
            )
            self.assertEqual([], build_fetches(context / "Dockerfile", context))

    def test_nvcr_is_only_a_base_image_registry(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            build = recipe["execution"]["build"]
            hosts = build["network"]["hosts"]
            if "nvcr.io" not in hosts:
                continue
            dockerfile = ROOT / build["dockerfile"]
            steps = "\n".join(
                line
                for line in dockerfile.read_text().splitlines()
                if not line.lstrip().upper().startswith("FROM ")
            )
            self.assertIn("https://nvcr.io", steps, path.name)

    def test_security_and_redirect_hosts_remain_explicit(self) -> None:
        for slug in (
            "ltx-2-19b-dev-fp4-pytorch-single",
            "step1x-3d-geometry-pytorch-single",
            "step1x-3d-label-geometry-pytorch-single",
            "step1x-3d-texture-pytorch-single",
        ):
            recipe = load(ROOT / "recipes" / f"{slug}.json")
            hosts = recipe["execution"]["build"]["network"]["hosts"]
            self.assertTrue(hosts, slug)
            if slug.startswith("step1x"):
                self.assertIn("security.ubuntu.com", hosts)

    def test_pytorch_index_object_host_is_explicit(self) -> None:
        for path in sorted((ROOT / "recipes").glob("*.json")):
            recipe = load(path)
            build = recipe["execution"]["build"]
            dockerfile = ROOT / build["dockerfile"]
            source = dockerfile.read_text(encoding="utf-8")
            if "https://download.pytorch.org/whl/" not in source:
                continue
            hosts = build["network"]["hosts"]
            self.assertIn("download.pytorch.org", hosts, path.name)
            self.assertIn("download-r2.pytorch.org", hosts, path.name)
            if any(
                f"https://download.pytorch.org/whl/{flavor}" in source
                for flavor in ("cu130", "cu132")
            ):
                self.assertIn("pypi.nvidia.com", hosts, path.name)


if __name__ == "__main__":
    unittest.main()
