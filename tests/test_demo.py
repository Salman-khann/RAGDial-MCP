from __future__ import annotations

from ragdial_mcp.demo import bar, main


def test_demo_runs_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    main(["--home", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    assert "Does header-aware chunking beat fixed-size windows?" in out
    assert "12 of 14" in out and "5 of 14" in out
    assert "markdown-256 wins" in out
    assert "5 questions now answered" in out and "none broken" in out
    assert "q01" in out and "What is the default size" in out


def test_bar_widths_are_exact():
    assert bar(0.0, 10, "red").plain == " " * 10
    assert bar(1.0, 10, "red").plain == "█" * 10
    assert bar(0.5, 10, "red").plain == "█████     "
    assert len(bar(0.73, 20, "red").plain) == 20
    assert bar(2.0, 4, "red").plain == "████"  # clamped
