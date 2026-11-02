from pathlib import Path

from support_rag.evaluation import evaluate


def test_labeled_offline_dataset(service):
    dataset = Path(__file__).resolve().parents[1] / "examples/evaluation.json"
    report = evaluate(service, dataset)
    assert report["passed"] == report["total"], report
