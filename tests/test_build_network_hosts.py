from __future__ import annotations

import glob
import json
import posixpath
import re
import shlex
import tempfile
import unittest
from pathlib import Path

from documents import recipe as document_recipe

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


INSTALL = re.compile(r"(?:\bpip3?|-m pip|\buv pip)\s+install\b(?P<args>[^&;|\n]*)")
ARGUMENT_FLAGS = {"-r", "-c", "-f", "--requirement", "--constraint", "--find-links"}


OTHER_FETCH = re.compile(
    r"(?:^|[\s;&|(])(?:curl|wget)\s"
    r"|\bgit\s+(?:clone|fetch|pull|submodule\s+update)\b"
    r"|\bapt(?:-get)?\s+(?:install|update|upgrade|download)\b"
    r"|\b(?:npm|pnpm|yarn)\s+(?:install|add|ci)\b"
    r"|\bcargo\s+(?:install|fetch)\b"
)
SCRIPT_SUFFIXES = {".sh", ".bash", ".py"}


def fetching_installs(text: str) -> list[str]:
    """Package or source fetches that may reach the network.

    pip/uv installs without --no-index and not only local paths, plus curl, wget, git clone,
    apt-get and npm-style installs. A pip install fetches nothing when --no-deps is set and
    every target is a local wheel, or a local source with --no-build-isolation (an isolated
    build downloads its build backend).
    """
    joined = re.sub(r"\\\n", " ", text)
    found: list[str] = []
    for match in INSTALL.finditer(joined):
        try:
            args = shlex.split(match.group("args"), posix=True)
        except ValueError:
            found.append(match.group(0).strip())
            continue
        if "--no-index" in args:
            continue
        targets: list[str] = []
        flags = iter(args)
        requirement_file = False
        for arg in flags:
            if arg in ARGUMENT_FLAGS:
                next(flags, None)
                requirement_file = requirement_file or arg in {"-r", "--requirement"}
            elif not arg.startswith("-"):
                targets.append(arg)
        local_only = (
            "--no-deps" in args
            and not requirement_file
            and bool(targets)
            and all(t.startswith(("/", ".")) for t in targets)
            and (
                "--no-build-isolation" in args
                or all(t.endswith(".whl") for t in targets)
            )
        )
        if not local_only:
            found.append(match.group(0).strip())
    for line in joined.splitlines():
        code = line.split("#", 1)[0] if line.lstrip().startswith("#") else line
        for match in OTHER_FETCH.finditer(code):
            found.append(match.group(0).strip(" ;&|("))
    return found


def _instructions(dockerfile_text: str) -> list[tuple[str, str]]:
    joined = re.sub(r"\\\n", " ", dockerfile_text)
    out: list[tuple[str, str]] = []
    for line in joined.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keyword, _, rest = line.partition(" ")
            out.append((keyword.upper(), rest.strip()))
    return out


def run_scripts(dockerfile_text: str, context: Path) -> dict[str, str]:
    """Context scripts the image build can execute, as {context-relative path: text}.

    A script is run when a RUN line names its in-image path (a `$var` or `*` in the RUN path
    stands for any single path segment, so `for mod in ...; do bash .../mods/$mod/run.sh`
    covers every mods/*/run.sh), or when a run script names a sibling file by its basename.
    """
    instructions = _instructions(dockerfile_text)
    workdir = "/"
    copied: dict[str, Path] = {}
    for keyword, rest in instructions:
        if keyword == "WORKDIR":
            workdir = posixpath.join(workdir, rest)
        if keyword not in {"COPY", "ADD"}:
            continue
        words = [w for w in rest.split() if not w.startswith("--")]
        if len(words) < 2 or "--from=" in rest or words[0].startswith("http"):
            continue
        *sources, dest = words
        dest = posixpath.join(workdir, dest)
        into_dir = dest.endswith("/") or len(sources) > 1
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
                        image = posixpath.join(dest, file.relative_to(base).as_posix())
                    elif into_dir:
                        image = posixpath.join(dest, file.name)
                    else:
                        image = dest
                    copied[posixpath.normpath(image)] = file
    run_text = "\n".join(rest for keyword, rest in instructions if keyword == "RUN")
    named = [t for t in re.findall(r"[\w$*{}./-]*/[\w$*{}./-]+", run_text)]
    templates = [
        re.compile(
            "^"
            + re.sub(r"\\\$\\\{\w+\\\}|\\\$\w+|\\\*", "[^/]+", re.escape(token))
            + "$"
        )
        for token in named
    ]
    executed = {
        image
        for image, file in copied.items()
        if file.suffix in SCRIPT_SUFFIXES and any(t.match(image) for t in templates)
    }
    pending = list(executed)
    while pending:
        text = copied[pending.pop()].read_text(encoding="utf-8", errors="replace")
        for image, file in copied.items():
            if (
                image not in executed
                and file.suffix in SCRIPT_SUFFIXES
                and posixpath.basename(image) in text
            ):
                executed.add(image)
                pending.append(image)
    return {
        copied[image].relative_to(context).as_posix(): copied[image].read_text(
            encoding="utf-8", errors="replace"
        )
        for image in sorted(executed)
    }


def build_fetches(dockerfile: Path, context: Path) -> list[str]:
    """Fetches in the Dockerfile and in the context scripts its RUN lines execute."""
    text = dockerfile.read_text(encoding="utf-8")
    found = fetching_installs(text)
    for name, script in run_scripts(text, context).items():
        found += [f"{name}: {hit}" for hit in fetching_installs(script)]
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

    def test_package_installs_declare_an_index_host(self) -> None:
        """The Spark builder gives an empty host list --network=none: a pip install would fail."""
        offline: dict[str, list[str]] = {}
        for path in sorted((ROOT / "recipes").glob("*.json")):
            build = document_recipe(path.stem).execution.build
            if not build.network.hosts:
                offline[path.stem] = build_fetches(
                    ROOT / build.dockerfile, ROOT / build.context.path
                )
        self.assertEqual({}, {k: v for k, v in offline.items() if v})

    def test_pip_fetches_declare_the_python_index_hosts(self) -> None:
        """A declared host list that omits the index fails the pip fetch just like an empty one."""
        missing: dict[str, list[str]] = {}
        for path in sorted((ROOT / "recipes").glob("*.json")):
            build = document_recipe(path.stem).execution.build
            hosts = set(build.network.hosts)
            if hosts >= {"pypi.org", "files.pythonhosted.org"}:
                continue
            pip = [
                hit
                for hit in build_fetches(
                    ROOT / build.dockerfile, ROOT / build.context.path
                )
                if "pip" in hit
            ]
            if pip:
                missing[path.stem] = pip
        self.assertEqual({}, missing)

    def test_fetching_install_detection(self) -> None:
        self.assertTrue(fetching_installs("RUN pip install foo"))
        self.assertTrue(fetching_installs("RUN python -m pip install -r /tmp/a.txt"))
        self.assertFalse(fetching_installs("RUN pip install --no-index /tmp/a.whl"))
        self.assertFalse(
            fetching_installs(
                "RUN python -m pip install --no-deps --no-build-isolation /tmp/src"
                " && pip show x"
            )
        )
        self.assertFalse(fetching_installs("RUN pip install --no-deps /tmp/a.whl"))
        # an isolated source build downloads its build backend
        self.assertTrue(fetching_installs("RUN pip install --no-deps /tmp/src"))
        # the arguments end at the line: a later --no-index or apostrophe is not ours
        self.assertTrue(fetching_installs("RUN pip install foo\nRUN echo --no-index"))
        self.assertTrue(fetching_installs("RUN pip install foo\n# it's a comment"))

    def test_script_fetch_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            context = Path(tmp)
            (context / "mods" / "a").mkdir(parents=True)
            (context / "mods" / "a" / "run.sh").write_text(
                "#!/bin/bash\nuv pip install --quiet soundfile\n", encoding="utf-8"
            )
            (context / "mods" / "a" / "helper.sh").write_text(
                "curl -fsSL https://example.com/x | sh\n", encoding="utf-8"
            )
            (context / "mods" / "a" / "unused.sh").write_text(
                "pip install nothing\n", encoding="utf-8"
            )
            dockerfile = (
                "FROM scratch\nCOPY mods /opt/build/mods\n"
                'RUN for m in a; do \\\n  bash "/opt/build/mods/$m/run.sh"; done\n'
            )
            (context / "mods" / "a" / "run.sh").write_text(
                "uv pip install --quiet soundfile\nbash helper.sh\n", encoding="utf-8"
            )
            scripts = run_scripts(dockerfile, context)
            self.assertEqual({"mods/a/run.sh", "mods/a/helper.sh"}, set(scripts))
            (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
            hits = build_fetches(context / "Dockerfile", context)
            self.assertEqual(2, len(hits), hits)
            # a script that installs nothing from the network is clean
            (context / "mods" / "a" / "run.sh").write_text(
                "echo ok\n", encoding="utf-8"
            )
            (context / "mods" / "a" / "helper.sh").write_text(
                "echo ok\n", encoding="utf-8"
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
