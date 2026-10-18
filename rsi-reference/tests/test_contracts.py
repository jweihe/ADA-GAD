import json
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from gad_rsi.contracts import evaluate, load_graph
from gad_rsi.suite import run
from gad_rsi.prepare import graph_bytes


def test_official_weibo_macos_sidecar_is_not_a_second_graph():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("weibo.pt", b"graph-payload")
        archive.writestr("__MACOSX/._weibo.pt", b"metadata")
    assert graph_bytes(stream.getvalue(), "weibo") == b"graph-payload"


def test_archive_with_another_graph_is_rejected():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("weibo.pt", b"graph-payload")
        archive.writestr("other.pt", b"unexpected-graph")
    with pytest.raises(ValueError, match="Unexpected graph"):
        graph_bytes(stream.getvalue(), "weibo")


def test_label_bearing_graph_is_rejected(tmp_path):
    path = tmp_path / "leaky.npz"
    np.savez(path, x=np.ones((4, 2)), edge_index=np.array([[0], [1]]), y=np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="exactly x and edge_index"):
        load_graph(path)


@pytest.mark.parametrize("bad", [np.array([0.1, np.nan, 0.2, 0.4]), np.array([0.1, 0.2])])
def test_invalid_scores_are_not_fabricated_rewards(tmp_path, bad):
    labels, scores = tmp_path / "labels.npz", tmp_path / "scores.npy"
    np.savez(labels, y=np.array([0, 0, 1, 1]))
    np.save(scores, bad)
    with pytest.raises(ValueError):
        evaluate(labels, scores, 4)


def test_ties_use_standard_auc_and_average_precision(tmp_path):
    y = np.array([0, 1, 0, 1, 0, 1])
    values = np.array([0.1, 0.4, 0.4, 0.7, 0.7, 0.7])
    np.savez(tmp_path / "labels.npz", y=y)
    np.save(tmp_path / "scores.npy", values)
    result = evaluate(tmp_path / "labels.npz", tmp_path / "scores.npy", len(y))
    assert result["roc_auc"] == roc_auc_score(y, values)
    assert result["average_precision"] == average_precision_score(y, values)


def test_sequential_run_resume_and_data_change_detection(tmp_path):
    data = tmp_path / "data"
    (data / "public").mkdir(parents=True)
    (data / "judge").mkdir()
    rng = np.random.default_rng(40)
    features = np.concatenate([rng.normal(0, 0.1, (40, 4)), rng.normal(6, 0.1, (8, 4))])
    edges = np.stack([np.arange(48), np.roll(np.arange(48), 1)])
    np.savez(data / "public" / "fixture.npz", x=features, edge_index=edges)
    np.savez(data / "judge" / "fixture.npz", y=np.array([0] * 40 + [1] * 8))
    plan = {"datasets": ["fixture"], "methods": ["iforest"], "seeds": [0, 1]}
    out = tmp_path / "runs"
    results = run(plan, out, data, timeout=60)
    assert all(r["status"] == "completed" for r in results)
    assert len(list(out.glob("fixture/iforest/*/attempt-*"))) == 2
    assert run(plan, out, data, timeout=60) == results
    assert len(list(out.glob("fixture/iforest/*/attempt-*"))) == 2
    np.savez(data / "judge" / "fixture.npz", y=np.array([1] * 8 + [0] * 40))
    with pytest.raises(ValueError, match="different input"):
        run(plan, out, data, timeout=60)


def test_multi_gpu_request_rejected_before_training(tmp_path):
    with pytest.raises(ValueError, match="exactly one GPU"):
        run({}, tmp_path / "unused", tmp_path, gpu="0,1")
