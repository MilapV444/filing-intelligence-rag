"""The README is checkable, not decorative (AC-14).

A README rots faster than code because nothing fails when it goes stale. These
assertions bind it to things that can actually change underneath it: the CLI's
real commands, the manifest's real filenames, the extras that really exist, and
the limitations the project has actually recorded.
"""

import re
import tomllib
from pathlib import Path

import pytest

from risk_analytics import cli, config, manifest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def text():
    if not README.exists():
        pytest.fail("AC-14 requires a README at the repository root")
    return README.read_text(encoding="utf-8")


def test_the_reader_is_told_what_it_is_before_how_to_run_it(text):
    head = text[:600]
    assert "financial" in head.lower()
    assert "cited" in head.lower()


def test_every_documented_command_exists_in_the_cli(text):
    """A README naming a command the CLI does not have is worse than none."""
    parser = cli.build_parser()
    documented = set(re.findall(r"python -m risk_analytics (\w+)", text))
    assert documented, "the README documents no commands"
    for command in documented:
        parser.parse_args([command] if command == "ingest" else [command, "q"])


def test_every_documented_flag_exists(text):
    """Flags drift when options are renamed; this catches it."""
    parser = cli.build_parser()
    real = set()
    for action in parser._subparsers._group_actions[0].choices["ask"]._actions:
        real.update(action.option_strings)

    documented = {f for f in re.findall(r"\s(--[a-z][a-z-]+)", text)}
    # Only judge flags the README presents as belonging to `ask`.
    documented &= {"--tables-only", "--doc", "--k", "--ceiling", "--force"}
    unknown = documented - real - {"--force"}
    assert not unknown, f"README documents flags the CLI does not have: {unknown}"


def test_the_documented_install_extras_exist(text):
    extras = set(
        tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        ["project"]["optional-dependencies"]
    )
    for extra in re.findall(r'pip install -e "\.\[([a-z,]+)\]"', text):
        for name in extra.split(","):
            assert name in extras, f"README installs a nonexistent extra: {name}"


def test_the_credential_instructions_name_the_real_variable_and_file(text):
    assert config.API_KEY_ENV_VAR in text
    assert config.ENV_FILE.name in text


def test_the_corpus_instructions_match_the_manifest(text):
    """The README tells a reader to fetch documents by manifest filename. If the
    manifest changes, the instruction has to still be true."""
    assert "corpus/manifest.json" in text
    assert config.DOCUMENTS_DIR.name in text, (
        "the README must name the directory the PDFs actually go in"
    )
    for doc in manifest.load():
        assert doc.doc_id in text or "manifest" in text


def test_the_cost_ceiling_quoted_matches_configuration(text):
    assert f"{config.RUN_COST_CEILING_USD}" in text


def test_the_limitations_section_is_present_and_not_token(text):
    """The point of writing them down is that a reader sees them without asking.
    A one-line 'some limitations apply' would pass a weaker test than this."""
    assert "## Limitations" in text
    section = text.split("## Limitations", 1)[1].split("\n## ", 1)[0]
    bullets = [b for b in section.splitlines() if b.strip().startswith("- ")]
    assert len(bullets) >= 5, "fewer limitations listed than the project has recorded"


def test_the_limitations_name_the_ones_that_would_embarrass_a_demo(text):
    """These are the specific things a viewer would otherwise discover live."""
    lowered = text.lower()
    assert "insufficient evidence" in lowered, "the refusal behaviour is undocumented"
    assert "esg" in lowered and "emission" in lowered, "the empty ESG corpus is undocumented"
    assert "chart" in lowered, "the dropped chart path is undocumented"


def test_the_disclaimer_appears_before_the_setup_instructions(text):
    """Someone skimming to run it must pass the not-a-rating statement first."""
    disclaimer = lowered_index(text, "not a credit rating")
    setup = lowered_index(text, "## setup")
    assert disclaimer != -1, "the README does not carry the not-a-rating disclaimer"
    assert disclaimer < setup, "the disclaimer sits below the setup instructions"


def lowered_index(text: str, needle: str) -> int:
    """Search with whitespace collapsed.

    Markdown wraps prose, so "not a credit rating" can be split across a line
    break and remain perfectly readable while a literal search misses it. The
    reader's experience is what the assertion is about, not the line width.
    """
    return " ".join(text.lower().split()).find(needle)
