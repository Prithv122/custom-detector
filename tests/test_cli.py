import json

import pytest

from detector.cli import build_parser


def test_summary_reports_the_committed_split(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["summary"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "images: {'train': 1263, 'val': 331, 'test': 402}" in out
    assert "cross-split near-duplicate pairs: 0" in out


def test_leak_audit_writes_the_published_split_counts(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "leak_audit.json"
    args = build_parser().parse_args(["leak-audit", "--out", str(out)])
    assert args.func(args) == 0
    assert "nearest_twin_other_split: 315" in capsys.readouterr().out
    assert json.loads(out.read_text())["any_twin_other_split"] == 676


def test_download_without_a_key_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    args = build_parser().parse_args(["download"])
    assert args.func(args) == 1
    assert "ROBOFLOW_API_KEY is not set" in capsys.readouterr().err
