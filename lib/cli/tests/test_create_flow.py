from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from click.testing import CliRunner
import pytest
from pytest import MonkeyPatch
import tomli

from crewai_cli.cli import crewai
from crewai_cli.create_flow import create_flow


def test_create_flow_declarative_project_can_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow", declarative=True)

    project_root = tmp_path / "research_flow"
    assert project_root.is_dir()

    pyproject = tomli.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["name"] == "research_flow"
    assert pyproject["project"]["requires-python"]
    assert pyproject["project"]["dependencies"]
    assert (project_root / pyproject["tool"]["crewai"]["definition"]).is_file()
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "CrewAI Flow declaration" in agents_md
    assert "schema: crewai.flow/v1" in agents_md
    assert "do not assemble the string with CEL `+`" in agents_md
    assert "Agent prompt template. Insert Flow values with `${...}`" in agents_md
    assert "Runtime inputs passed to the Crew" in agents_md
    assert "call: expression" in agents_md
    assert "call: tool" not in agents_md
    assert "call: script" not in agents_md
    assert "call: each" not in agents_md
    assert "human_feedback" not in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    cursor_md = (project_root / "CURSOR.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in cursor_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()

    monkeypatch.chdir(project_root)
    result = CliRunner().invoke(crewai, ["run"], env={"UV_RUN_RECURSION_DEPTH": "1"})

    assert result.exit_code == 0
    assert "Running the Flow" not in result.output
    assert "AI agents" in result.output


def test_create_flow_scaffolds_assistant_instructions(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_md.index("## Working in this project") < agents_md.index("## Reference")
    assert "Never turn off tracing or telemetry" in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()


def test_scaffolded_flow_agents_md_tells_assistants_to_evaluate_with_crewai_eval(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")
    create_flow("Declared Flow", declarative=True)

    python_flow = (tmp_path / "research_flow" / "AGENTS.md").read_text(encoding="utf-8")
    assert "is done when a traced run passes `crewai eval`" in python_flow
    assert "prints a markdown brief of what failed and the change to make" in python_flow
    assert "`crewai eval --models " in python_flow
    # The declarative flow ships its own AGENTS.md.
    declared = (tmp_path / "declared_flow" / "AGENTS.md").read_text(encoding="utf-8")
    assert "Run `crewai eval` after a change" in declared
    assert 'crewai eval --models "provider/model,…"' in declared
    for agents_md in (python_flow, declared):
        assert "{{name}}" not in agents_md
        assert "crewai-eval:" not in agents_md


@pytest.fixture
def scaffolded_flow_main(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> Iterator[ModuleType]:
    """The real `crewai create flow` main.py, with its crew replaced by a fake that
    echoes the topic, so the flow runs end to end without an LLM."""
    monkeypatch.chdir(tmp_path)
    create_flow("Demo Flow")
    monkeypatch.chdir(tmp_path / "demo_flow")
    monkeypatch.syspath_prepend(str(tmp_path / "demo_flow" / "src"))

    main = importlib.import_module("demo_flow.main")

    class FakeCrew:
        def kickoff(self, inputs: dict[str, str]) -> SimpleNamespace:
            return SimpleNamespace(raw=f"post about {inputs['topic']}")

    class FakeContentCrew:
        def crew(self) -> FakeCrew:
            return FakeCrew()

    monkeypatch.setattr(main, "ContentCrew", FakeContentCrew)
    yield main
    # Only the project's own modules: re-importing a native extension such as
    # lancedb after dropping it from sys.modules panics.
    for name in [m for m in sys.modules if m.split(".")[0] == "demo_flow"]:
        del sys.modules[name]


def _written_post() -> str:
    return Path("output/post.md").read_text(encoding="utf-8")


def test_scaffolded_flow_kickoff_runs_with_json_inputs_from_argv(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["kickoff", '{"topic": "Quantum"}'])

    scaffolded_flow_main.kickoff()

    assert _written_post() == "post about Quantum"


def test_scaffolded_flow_kickoff_without_inputs_uses_default_topic(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["kickoff"])

    scaffolded_flow_main.kickoff()

    assert _written_post() == "post about AI Agents"


@pytest.mark.parametrize(
    ("argv_inputs", "message"),
    [
        ("{bad", r"Invalid inputs JSON: Expecting property name"),
        ("[1]", r"Invalid inputs: expected a JSON object\."),
        ('"Quantum"', r"Invalid inputs: expected a JSON object\."),
    ],
)
def test_scaffolded_flow_kickoff_rejects_inputs_that_are_not_a_json_object(
    scaffolded_flow_main: ModuleType,
    monkeypatch: MonkeyPatch,
    argv_inputs: str,
    message: str,
):
    monkeypatch.setattr(sys, "argv", ["kickoff", argv_inputs])

    with pytest.raises(SystemExit, match=message) as exc_info:
        scaffolded_flow_main.kickoff()

    assert """uv run kickoff '{"topic": "AI Agents"}'""" in str(exc_info.value)
    assert not Path("output/post.md").exists()


def test_scaffolded_flow_start_step_keeps_inputs_passed_to_flow_kickoff(
    scaffolded_flow_main: ModuleType,
):
    # AMP calls `YourFlow().kickoff(inputs=...)` directly, not main.py's kickoff().
    scaffolded_flow_main.ContentFlow().kickoff(inputs={"topic": "Robots"})

    assert _written_post() == "post about Robots"


def test_scaffolded_flow_run_with_trigger_uses_payload_topic(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["run_with_trigger", '{"topic": "Rockets"}'])

    scaffolded_flow_main.run_with_trigger()

    assert _written_post() == "post about Rockets"


def test_scaffolded_flow_run_with_trigger_without_topic_keeps_default(
    scaffolded_flow_main: ModuleType, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(sys, "argv", ["run_with_trigger", '{"source": "slack"}'])

    scaffolded_flow_main.run_with_trigger()

    assert _written_post() == "post about AI Agents"


def test_scaffolded_flow_docs_tell_how_to_pass_inputs(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Demo Flow")

    agents_md = (tmp_path / "demo_flow" / "AGENTS.md").read_text(encoding="utf-8")
    readme = (tmp_path / "demo_flow" / "README.md").read_text(encoding="utf-8")
    assert "`uv run kickoff '{\"topic\":\"...\"}'`" in agents_md
    assert "uv run kickoff '{\"topic\": \"Quantum Computing\"}'" in readme
