from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_phase0_docs_and_execute_log_exist_with_expected_sections():
    docs_dir = REPO_ROOT / "docs"
    assert docs_dir.is_dir()

    problem_definition = docs_dir / "problem_definition.md"
    io_spec = docs_dir / "io_spec.md"
    metrics_spec = docs_dir / "metrics_spec.md"
    execute_log = REPO_ROOT / "EXECUTE_LOG.md"

    for path in (problem_definition, io_spec, metrics_spec, execute_log):
        assert path.is_file(), str(path)

    assert "Research Goal" in problem_definition.read_text(encoding="utf-8")
    assert "Single-Vehicle Dataset Interface" in io_spec.read_text(encoding="utf-8")
    assert "Platoon Metrics" in metrics_spec.read_text(encoding="utf-8")

    log_text = execute_log.read_text(encoding="utf-8")
    assert "Project Snapshot" in log_text
    assert "Current Status" in log_text
    assert "Phase Checklist" in log_text
    assert "Task Log" in log_text
    assert "Open Decisions" in log_text
    assert "Handoff Notes" in log_text


def test_agents_marks_phase0_as_completed():
    agents_text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "- [x] 达到Phase 0验收标准" in agents_text
