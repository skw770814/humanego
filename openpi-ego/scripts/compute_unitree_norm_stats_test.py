import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openpi.policies import unitree_robot_policy
from openpi.shared import normalize
from scripts import compute_unitree_norm_stats as norm_stats


def _eef18(count: int) -> np.ndarray:
    values = np.zeros((count, 18), dtype=np.float32)
    identity = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    values[:, 3:9] = identity
    values[:, 12:18] = identity
    values[:, 0] = np.arange(count, dtype=np.float32) * 0.01
    values[:, 9] = np.arange(count, dtype=np.float32) * -0.01
    return values


def test_eef18_only_canonicalization_and_relative_chunks():
    states = _eef18(4)
    actions = states.copy()
    actions[:, 0] += 0.02
    actions[:, 9] -= 0.03

    canonical_states = norm_stats._canonicalize(states, "columns_grouped", "pelvis_wrist")  # noqa: SLF001
    canonical_actions = norm_stats._canonicalize(actions, "columns_grouped", "pelvis_wrist")  # noqa: SLF001
    chunks, valid = norm_stats._make_action_chunks(  # noqa: SLF001
        canonical_states,
        canonical_actions,
        anchor_count=4,
        horizon=3,
        mode="shared",
    )

    assert canonical_states.shape == (4, 18)
    assert chunks.shape == (4, 3, 18)
    assert valid.shape == (4, 3)
    np.testing.assert_allclose(chunks[0, :, 0], [0.02, 0.03, 0.04], atol=1e-6)
    np.testing.assert_allclose(chunks[0, :, 9], [-0.03, -0.04, -0.05], atol=1e-6)


@pytest.mark.parametrize(
    "input_frame",
    ["native", "torso_palm", "g1_base_tcp", "recording_tcp", "pelvis_wrist"],
)
def test_norm_and_training_human_frame_canonicalization_are_identical(input_frame):
    values = _eef18(4)

    expected = norm_stats._canonicalize(values, "columns_grouped", input_frame)  # noqa: SLF001
    actual = unitree_robot_policy.CanonicalizeHumanEEF(
        action_dim=18,
        input_frame=input_frame,
    )({"state": values[0].copy(), "actions": values.copy()})

    np.testing.assert_allclose(actual["state"], expected[0], atol=1e-7)
    np.testing.assert_allclose(actual["actions"], expected, atol=1e-7)


def test_eef18_only_hybrid_is_eef_per_step_without_a_tail():
    accumulator = norm_stats.ActionAccumulator(
        mode="hybrid",
        horizon=2,
        vector_dim=18,
        reservoir_size=128,
        seed=7,
    )
    chunks = np.stack((_eef18(2), _eef18(2) + 0.01))
    # Restore valid rotations after adding the test offset.
    chunks[..., 3:9] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    chunks[..., 12:18] = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)
    accumulator.update(chunks, np.ones((2, 2), dtype=np.bool_))

    stats, minimum, maximum, details = accumulator.finalize()

    assert stats.mean.shape == (2, 18)
    assert minimum.shape == maximum.shape == (2, 18)
    assert details["hybrid_eef_only"] is True


def test_eef_only_projection_drops_nonzero_gripper_values_before_statistics():
    source = np.concatenate(
        (
            _eef18(4),
            np.asarray([[0.9, -0.8], [0.7, -0.6], [0.5, -0.4], [0.3, -0.2]], dtype=np.float32),
        ),
        axis=-1,
    )

    projected = norm_stats._select_eef_only(source, "action")  # noqa: SLF001

    assert projected.shape == (4, 18)
    np.testing.assert_array_equal(projected, source[:, :18])


def test_per_step_stats_report_actionable_error_for_too_short_episodes():
    accumulator = norm_stats.ActionAccumulator(
        mode="per_step",
        horizon=3,
        vector_dim=18,
        reservoir_size=128,
        seed=11,
    )
    chunks = np.stack((_eef18(3), _eef18(3)))
    accumulator.update(chunks, np.asarray([[True, False, False], [True, False, False]]))

    with pytest.raises(ValueError, match="longer episodes"):
        accumulator.finalize()


def test_per_step_normalization_reports_missing_horizon_samples():
    accumulator = norm_stats.ActionAccumulator(
        mode="per_step",
        horizon=3,
        vector_dim=18,
        reservoir_size=128,
        seed=11,
    )
    accumulator.update(
        np.stack((_eef18(3), _eef18(3))),
        np.asarray([[True, True, False], [True, True, False]]),
    )

    with np.testing.assert_raises_regex(ValueError, "insufficient indices=\\[2\\]"):
        accumulator.finalize()


def test_eef20_source_writes_eef18_assets_for_all_norm_modes_and_progress(tmp_path):
    dataset = tmp_path / "human_eef20"
    data_dir = dataset / "data/chunk-000"
    meta_dir = dataset / "meta"
    data_dir.mkdir(parents=True)
    meta_dir.mkdir()

    frame_count = 60
    eef = _eef18(frame_count)
    gripper = np.stack(
        (
            np.linspace(0.2, 0.9, frame_count, dtype=np.float32),
            np.linspace(-0.8, -0.1, frame_count, dtype=np.float32),
        ),
        axis=-1,
    )
    states = np.concatenate((eef, gripper), axis=-1)
    actions = states.copy()
    actions[:, 0] += 0.01
    vector_type = pa.list_(pa.float32(), 20)
    pq.write_table(
        pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=vector_type),
                "action": pa.array(actions.tolist(), type=vector_type),
            }
        ),
        data_dir / "episode_000000.parquet",
    )

    names = [*[f"eef_{index}" for index in range(18)], "kLeftGripper", "kRightGripper"]
    feature = {"dtype": "float32", "shape": [20], "names": [names]}
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "robot_type": "human",
                "fps": 30,
                "chunks_size": 1000,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "features": {"observation.state": feature, "action": feature},
            }
        )
    )
    (meta_dir / "episodes.jsonl").write_text(json.dumps({"episode_index": 0, "length": frame_count}) + "\n")

    for mode in ("absolute", "shared", "per_step", "hybrid"):
        output = tmp_path / f"stats_{mode}"
        norm_stats.main(
            [
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output),
                "--asset-id",
                f"human_eef20_projected_to_eef18_{mode}",
                "--norm-mode",
                mode,
                "--action-horizon",
                "10",
                "--action-source-step-scale",
                "0.5",
                "--input-rotation-format",
                "columns_grouped",
                "--input-frame",
                "pelvis_wrist",
                "--select-eef-only",
                "--episodes",
                "0",
                "--reservoir-size",
                "128",
            ]
        )

        stats = normalize.load(output)
        manifest = json.loads((output / "norm_stats_manifest.json").read_text())
        expected_action_shape = (18,) if mode in ("absolute", "shared") else (10, 18)
        assert stats["state"].mean.shape == (18,)
        assert stats["actions"].mean.shape == expected_action_shape
        assert manifest["canonical_layout"]["dimension"] == 18
        assert manifest["canonical_layout"]["action_domain"] == "eef_only"
        assert manifest["action_chunk_resampling"]["enabled"] is True
        assert manifest["action_chunk_resampling"]["source_step_scale"] == 0.5
        assert manifest["source_projection"] == {
            "input_dimension": 20,
            "output_dimension": 18,
            "kept_slice": [0, 18],
            "ignored_slice": [18, 20],
            "ignored_tail": "gripper2",
            "applied_before": [
                "frame_canonicalization",
                "relative_action_conversion",
                "task_progress_resampling",
                "normalization_accumulation",
            ],
        }
