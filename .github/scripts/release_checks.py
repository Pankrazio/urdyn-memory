"""Release checks for built distributions, installed packages and pytest reports."""

from __future__ import annotations

import email.parser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import xml.etree.ElementTree as ET
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_release(repository: Path):
    """Only an exact, public-version tag at origin/main's tip may publish.

    Branch and tag protection are external prerequisites: Git ancestry alone
    cannot establish who reviewed or authorized a revision.
    """
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref = os.environ.get("GITHUB_REF", "")
    authorized = False
    if event == "push" and ref.startswith("refs/tags/v"):
        from packaging.version import Version

        version = tomllib.loads((repository / "pyproject.toml").read_text())["project"]["version"]
        parsed = Version(version)
        require(parsed.local is None, "Local package versions cannot be published to PyPI")
        require(str(parsed) == version, "The release version must use canonical spelling")
        require(ref == f"refs/tags/v{version}", "Release tag and package version differ")

        def commit(reference):
            return subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "--verify", reference + "^{commit}"],
                text=True,
            ).strip()

        head = commit("HEAD")
        require(commit(ref) == head, "Release tag does not identify the checked-out revision")
        require(commit(os.environ["GITHUB_SHA"]) == head, "Release event and checked-out revision differ")
        require(commit("refs/remotes/origin/main") == head,
                "Release revision must be the exact tip of origin/main")
        authorized = True
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a") as stream:
            stream.write(f"authorized={str(authorized).lower()}\n")
    print("Release tag/version/revision checks passed." if authorized else "Verification only: publication is disabled.")


def distributions(directory: Path):
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Expected exactly one wheel and one sdist")
    return wheels[0], sdists[0]


def wheel_sources(wheel: Path):
    with zipfile.ZipFile(wheel) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()
                if info.filename.startswith("urdyn/") and not info.is_dir()}


def check_distributions(directory: Path, repository: Path, output: Path):
    wheel, sdist = distributions(directory)
    project = tomllib.loads((repository / "pyproject.toml").read_text())["project"]
    sources = {f.relative_to(repository / "src").as_posix(): f.read_bytes()
               for f in (repository / "src/urdyn").rglob("*")
               if f.is_file() and "__pycache__" not in f.parts}
    require(wheel_sources(wheel) == sources, "Wheel contents differ from the checked-out engine")
    with zipfile.ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        require(len(names) == 1, "Expected one wheel metadata record")
        wheel_metadata = email.parser.BytesParser().parsebytes(archive.read(names[0]))
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        prefixes = {member.name.split("/", 1)[0] for member in members}
        require(len(prefixes) == 1, "Expected a single sdist root")
        prefix = prefixes.pop() + "/"
        packaged = {member.name[len(prefix + "src/"):]: archive.extractfile(member).read()
                    for member in members if member.isfile() and member.name.startswith(prefix + "src/urdyn/")}
        require(packaged == sources, "Sdist contents differ from the checked-out engine")
        # The build backend normalizes TOML formatting in the sdist.
        require(tomllib.loads(archive.extractfile(prefix + "pyproject.toml").read().decode())
                == tomllib.loads((repository / "pyproject.toml").read_text()),
                "Sdist project metadata differs from the checkout")
        sdist_metadata = email.parser.BytesParser().parsebytes(archive.extractfile(prefix + "PKG-INFO").read())
    for metadata in [wheel_metadata, sdist_metadata]:
        require(metadata["Name"] == project["name"] and metadata["Version"] == project["version"],
                "Distribution name/version differs from project metadata")
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in [wheel, sdist]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())))
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    provenance = {"commit": commit, "version": project["version"], "artifacts": hashes,
                  "uv_lock_sha256": hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Wheel and sdist match the checked-out engine; checksums recorded.")


def check_report(report: Path, expected_path: Path):
    expected = json.loads(expected_path.read_text())
    root = ET.parse(report).getroot()
    cases = root.findall(".//testcase")
    ids = [(case.get("classname"), case.get("name")) for case in cases]
    require(len(ids) == len(set(ids)), "Duplicate test results")
    require(not any(case.find("failure") is not None or case.find("error") is not None for case in cases),
            "The suite contains failures or errors")
    require(not any(int(suite.get("errors", "0")) or int(suite.get("failures", "0"))
                    for suite in root.iter("testsuite")), "The suite contains collection errors")
    skipped = [case for case in cases if case.find("skipped") is not None]
    actual = sorted([case.get("classname"), case.get("name")] for case in skipped)
    require(len(actual) == 59 and actual == expected["skipped_tests"], "Unexpected or missing ONNX skips")
    require(all(case.find("skipped").get("message") == expected["reason"] for case in skipped),
            "A skipped test has an unexpected reason")
    passed = len(cases) - len(skipped)
    require(passed >= expected["minimum_passed"], "The public suite is incomplete")
    print(f"{passed} passed; 59 expected real-ONNX tests NOT executed; no other skips.")


def smoke(directory: Path):
    import urdyn
    from urdyn import Urdyn, backup

    require(sys.prefix != sys.base_prefix, "A clean virtual environment is required")
    package = Path(urdyn.__file__).resolve().parent
    require(package.is_relative_to(Path(sys.prefix).resolve()), "Imported engine is outside the virtual environment")
    wheel, _ = distributions(directory)
    for name, content in wheel_sources(wheel).items():
        require((package.parent / name).read_bytes() == content, "Installed engine differs from the release wheel")
    cli = Path(sys.executable).with_name("urdyn")
    with tempfile.TemporaryDirectory(prefix="urdyn-release-smoke-") as temporary:
        base = Path(temporary)
        root = base / "project"
        root.mkdir()
        subprocess.run([str(cli), "create", "general"], cwd=root, check=True, capture_output=True, text=True)
        brain = Urdyn.open(root)
        original = brain.remember("Use SQLite for persistent project memory.", kind="decision")
        current = brain.remember("Use SQLite with explicit backups.", kind="decision", supersedes=original.memory_id)
        brain.record_attempt(task="migrate SQLite", approach="direct migration", outcome="failed")
        lesson = brain.learn("Back up SQLite before migrations.")
        skill = brain.promote(lesson, name="Safe migration", purpose="Preserve SQLite memory", steps=["back up", "migrate"])
        require(len(brain.timeline()) == 3, "Smoke history is incomplete")
        require(original.memory_id not in {m.memory_id for m in brain.state()}, "Supersession was lost")
        require(brain.get_skill(skill.skill_id) == skill, "Skill did not persist")
        brain.preflight("migrate SQLite")
        brain.context("migrate SQLite").render()
        brain.guard("migrate SQLite")
        require(brain.rebuild_search_index().status == "usable", "FTS rebuild failed")
        for args in [["status"], ["recall", "SQLite"], ["search-index", "status"]]:
            subprocess.run([str(cli), *args], cwd=root, check=True, capture_output=True, text=True)
        archive = base / "backup.zip"
        backup.create_backup(root, archive)
        backup.inspect_backup(archive)
        restored = base / "restored"
        backup.restore_backup(restored, archive)
        reopened = Urdyn.open(restored)
        require(reopened.timeline() == brain.timeline(), "Backup/restore changed canonical history")
        require(current.memory_id in {m.memory_id for m in reopened.state()}, "Restored memory is missing")
        require(reopened.get_skill(skill.skill_id) == skill, "Backup/restore changed the skill")
    print("Installed-wheel API, CLI, FTS, history and backup/restore smoke passed.")


if __name__ == "__main__":
    command, *arguments = sys.argv[1:]
    functions = {"release": check_release, "distributions": check_distributions, "report": check_report, "smoke": smoke}
    functions[command](*(Path(argument).resolve() for argument in arguments))
