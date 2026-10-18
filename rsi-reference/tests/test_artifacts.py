import io
import zipfile

import numpy as np
import pytest

from gad_rsi.artifacts import load_weights, write_weights


def test_weights_roundtrip_has_exact_state_names_and_values(tmp_path):
    expected = {"encoder.weight": np.zeros((2, 3), dtype=np.float32),
                "steps": np.array(4, dtype=np.int64)}
    state = {"encoder.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
             "steps": np.array(7, dtype=np.int64)}
    path = tmp_path / "weights.npz"
    write_weights(path, state)
    actual = load_weights(path, expected)
    assert set(actual) == set(state)
    for key in state:
        np.testing.assert_array_equal(actual[key], state[key])
        assert actual[key].dtype == state[key].dtype


@pytest.mark.parametrize("value", [
    np.zeros((100, 3), dtype=np.float32),
    np.zeros((2, 3), dtype=np.float64),
    np.full((2, 3), np.nan, dtype=np.float32),
    np.full((2, 3), np.inf, dtype=np.float32),
    np.full((2, 3), {"unexpected": "pickle"}, dtype=object),
    np.asfortranarray(np.zeros((2, 3), dtype=np.float32)),
])
def test_candidate_cannot_change_model_schema_or_supply_nonfinite_weights(tmp_path, value):
    path = tmp_path / "bad.npz"
    np.savez(path, weight=value)
    with pytest.raises(ValueError):
        load_weights(path, {"weight": np.zeros((2, 3), dtype=np.float32)})


@pytest.mark.parametrize("members", [["other.npy"], ["weight.npy", "weight.npy"],
                                      ["weight.npy", "../extra.npy"], []])
def test_unknown_missing_and_duplicate_members_are_rejected(tmp_path, members):
    path = tmp_path / "bad.npz"
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(2, dtype=np.float32), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as archive:
        for name in members:
            archive.writestr(name, buffer.getvalue())
    with pytest.raises(ValueError, match="names"):
        load_weights(path, {"weight": np.zeros(2, dtype=np.float32)})


def test_forged_enormous_shape_is_rejected_before_payload_allocation(tmp_path):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(buffer, {
        "descr": "<f4", "fortran_order": False, "shape": (10**15,),
    })
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("weight.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="shape"):
        load_weights(path, {"weight": np.zeros(2, dtype=np.float32)})


def test_compressed_oversized_member_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("weight.npy", b"\0" * 1_000_000)
    with pytest.raises(ValueError, match="oversized"):
        load_weights(path, {"weight": np.zeros(2, dtype=np.float32)})


def test_trailing_array_payload_is_rejected(tmp_path):
    buffer = io.BytesIO()
    np.save(buffer, np.zeros(2, dtype=np.float32), allow_pickle=False)
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("weight.npy", buffer.getvalue() + b"extra")
    with pytest.raises(ValueError, match="size"):
        load_weights(path, {"weight": np.zeros(2, dtype=np.float32)})


def test_symlink_is_not_followed(tmp_path):
    path = tmp_path / "weights.npz"
    write_weights(path, {"weight": np.zeros(2, dtype=np.float32)})
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(OSError):
        load_weights(link, {"weight": np.zeros(2, dtype=np.float32)})
