import re
from pathlib import Path

from tests.conftest import manifest_version


REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"


# fork: better-hermes-lcm — this was the literal "1.0.0-rc.1" while plugin.yaml had moved to
# 1.1.0-beta.2, so every assertion below tested a version nothing in the tree shipped any more
# and the release-cut gate had been red for two releases. Deriving it from the manifest is what
# stops the same drift happening at the next cut; the assertions themselves are unchanged.
RELEASE_VERSION = manifest_version()
RELEASE_NOTES = REPO_ROOT / ".github" / "release-notes" / f"v{RELEASE_VERSION}.md"


def test_release_workflow_requires_curated_tag_specific_notes():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert 'NOTES_FILE=".github/release-notes/${GITHUB_REF_NAME}.md"' in workflow
    assert 'body_path: .github/release-notes/${{ github.ref_name }}.md' in workflow
    assert "generate_release_notes: false" in workflow
    assert "git log --pretty" not in workflow


def test_release_workflow_marks_rc_tags_as_prereleases_not_latest():
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

    assert "prerelease: ${{ contains(github.ref_name, '-') }}" in workflow
    assert "make_latest: ${{ contains(github.ref_name, '-') && 'false' || 'true' }}" in workflow
    assert "draft: false" in workflow


def test_release_candidate_identity_surfaces_are_synchronized():
    manifest = (REPO_ROOT / "plugin.yaml").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    operator_guide = (REPO_ROOT / "docs" / "operator-guide.md").read_text(
        encoding="utf-8"
    )
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    bug_report = (
        REPO_ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.yml"
    ).read_text(encoding="utf-8")

    assert f"version: {RELEASE_VERSION}" in manifest
    assert f"hermes-lcm v{RELEASE_VERSION} (15 tools)" in readme
    assert f"hermes-lcm v{RELEASE_VERSION} (15 tools)" in operator_guide
    assert f"## v{RELEASE_VERSION} - " in changelog
    assert f"v{RELEASE_VERSION}, main, or commit SHA" in bug_report


def test_upgrade_guide_requires_sqlite_safe_backup_semantics():
    operator_guide = " ".join(
        (REPO_ROOT / "docs" / "operator-guide.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "the only supported online backup path" in operator_guide
    assert "stop Hermes and every other process that can write the database" in operator_guide
    assert "`lcm.db-wal` and `lcm.db-shm`" in operator_guide
    assert "one quiescent snapshot" in operator_guide


def test_upgrade_guide_covers_stable_and_prerelease_paths():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    operator_guide = " ".join(
        (REPO_ROOT / "docs" / "operator-guide.md")
        .read_text(encoding="utf-8")
        .split()
    )

    # NOT derived from plugin.yaml, deliberately: this heading names the release the guide
    # documents upgrading TO from the two older ones, so it is a historical fact about a past
    # cut, not a statement of what this checkout runs. Deriving it would rewrite accurate
    # upgrade documentation every time the manifest moves.
    assert "## Upgrade from v0.20.0 or v0.21.0-rc2 to v1.0.0-rc.1" in operator_guide
    assert (
        "A database created by either v0.20.0 or v0.21.0-rc2 opens in place"
        in operator_guide
    )
    assert (
        "For rollback to either v0.20.0 or v0.21.0-rc2, restore the pre-upgrade "
        "backup" in operator_guide
    )
    assert (
        "docs/operator-guide.md#upgrade-from-v0200-or-v0210-rc2-to-v100-rc1"
        in readme
    )


def test_preanswer_guide_discloses_inherited_embedding_provider_behavior():
    operator_guide = " ".join(
        (REPO_ROOT / "docs" / "operator-guide.md")
        .read_text(encoding="utf-8")
        .split()
    )

    assert "may call `lcm_recall`" in operator_guide
    assert "may send the current question to that provider" in operator_guide
    assert "Disabling the selective compiler does not prevent" in operator_guide
    assert "Pre-answer evidence alone remains provider-free." not in operator_guide


def _released_versions_newest_first() -> list[str]:
    """The versions CHANGELOG.md records, in the order it records them (newest first)."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.findall(r"^## (v[0-9A-Za-z.\-]+)", changelog, re.MULTILINE)


def _assert_notes_cover_only_their_own_release_scope(path: Path) -> None:
    notes = path.read_text(encoding="utf-8")
    version = path.stem  # "v1.1.0-beta.2"
    released = _released_versions_newest_first()

    # the first line names exactly this release, and nothing else
    first_line = notes.splitlines()[0]
    assert first_line.startswith("# "), f"{path.name}: no title heading"
    assert first_line.endswith(f" {version}"), f"{path.name}: title is {first_line!r}"

    # it cites no release NEWER than itself. CHANGELOG.md is the ordering: a token that sits
    # ABOVE this release in it is a later one, and a token that is not recorded at all is a
    # release that was never cut. Either way the notes would be claiming scope they do not have.
    assert version in released, f"{path.name}: CHANGELOG.md has no entry for {version}"
    own_position = released.index(version)
    for token in sorted(set(re.findall(r"v[0-9]+\.[0-9]+\.[0-9]+[0-9A-Za-z.\-]*", notes))):
        assert token in released, f"{path.name} cites {token}, which CHANGELOG.md never recorded"
        assert released.index(token) >= own_position, (
            f"{path.name} cites {token}, a release later than {version}"
        )

    # curated prose, not the generator's output: `generate_release_notes: false` is asserted
    # above on the workflow, and these are the two markers its output would leave behind.
    assert "**Full Changelog**" not in notes, f"{path.name}: generated changelog block"
    assert "Merge pull request #" not in notes, f"{path.name}: merge-commit dump"
    assert notes.count("\n## ") >= 2, f"{path.name}: no sections, so nothing is scoped"


def test_release_candidate_notes_cover_only_the_merged_release_scope():
    # fork: better-hermes-lcm — this asserted the CONTENT of upstream's v1.0.0-rc.1 notes:
    # "#526", "#557", "#570", "c368323", the "## Highlights"/"## Changes"/"## Contributors"
    # headings, the words "release candidate", "disabled by default" and "rollback-journal", a
    # "# hermes-lcm v…" first line, and a 60-line bound. This fork never cut that release; its
    # own notes open "# better-hermes-lcm v…", carry different sections, and v1.1.0-beta.1.md is
    # 115 lines — so the 60-line bound would have rejected a release this project actually
    # shipped. The pin also pointed RELEASE_NOTES at a file that does not exist, which is why
    # the check has been failing rather than gating anything. What survives is the property the
    # test is named for — notes that cover only their own release's scope — asserted on every
    # shipped notes file rather than on one release's wording.
    assert RELEASE_NOTES.exists(), (
        f"no release notes for the version plugin.yaml declares: {RELEASE_NOTES}"
    )
    notes_files = sorted((REPO_ROOT / ".github" / "release-notes").glob("v*.md"))
    assert notes_files, "no release notes at all"
    for path in notes_files:
        _assert_notes_cover_only_their_own_release_scope(path)
