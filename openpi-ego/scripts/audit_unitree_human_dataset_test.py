import json
import pathlib

from scripts import audit_unitree_human_dataset


def _write_dataset(dataset: pathlib.Path, mode: str, tcp_semantics: str) -> None:
    (dataset / "meta").mkdir(parents=True)
    feature = {"shape": [20], "names": [[*[f"eef_{index}" for index in range(18)], "left", "right"]]}
    (dataset / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": feature, "action": feature}})
    )
    (dataset / "meta/action_semantics.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "tcp_semantics": tcp_semantics,
                "tcp_pose9": {
                    "rotation_6d": "first two columns of a 3x3 rotation matrix",
                },
            }
        )
    )


def test_audit_mode2_recommends_only_relative_mixing(tmp_path: pathlib.Path):
    _write_dataset(tmp_path, "mode2_state", "absolute_next_target_in_recording_frame")

    audit = audit_unitree_human_dataset.audit_dataset(tmp_path, "mode2")

    assert audit["contract"]["stored_action"] == "absolute"
    assert audit["recommendations"]["human_only"]["absolute_or_relative"] == "HUMAN_INPUT_FRAME=recording_tcp"
    assert audit["recommendations"]["robot_human_mixed"]["absolute"].startswith("unsupported")
