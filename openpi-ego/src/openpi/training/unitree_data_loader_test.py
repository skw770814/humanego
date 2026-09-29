import dataclasses
import json

import numpy as np
import pytest

from openpi import transforms
from openpi.training import data_loader
from openpi.training import unitree_train_config


class _ListDataset:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, index):
        return self.values[index]

    def __len__(self):
        return len(self.values)


def test_episode_frame_subset_handles_noncontiguous_global_ranges():
    subset = data_loader.EpisodeFrameSubset(_ListDataset(range(20)), [(2, 5), (10, 14)])
    assert len(subset) == 7
    assert [subset[index] for index in range(len(subset))] == [2, 3, 4, 10, 11, 12, 13]


def test_weighted_mixture_sampler_has_exact_group_ratio():
    datasets = [_ListDataset(range(20)), _ListDataset(range(8))]
    mixture = data_loader.WeightedMixtureDataset(datasets, [0.5, 0.5], ["robot", "human"])
    sampler = data_loader.WeightedMixtureBatchSampler(mixture, 8, seed=7, shuffle=True)
    batch = next(iter(sampler))
    component_counts = [0, 0]
    for index in batch:
        component = int(np.searchsorted(mixture.offsets, index, side="right") - 1)
        component_counts[component] += 1
    assert component_counts == [4, 4]


def test_robot_and_one_human_mode_have_one_to_one_ratio():
    quotas = data_loader.mixture_batch_quotas(32, np.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(quotas, [16, 16])


def test_robot20_and_human18_reference_states_collate_after_common_padding():
    pad = transforms.PadStatesAndActions(32, mask_padded_action_dims=True)
    samples = [
        pad(
            {
                "state": np.zeros(dimension, dtype=np.float32),
                "action_reference_state": np.zeros(dimension, dtype=np.float32),
                "actions": np.zeros((50, dimension), dtype=np.float32),
            }
        )
        for dimension in (20, 18)
    ]

    batch = data_loader._collate_fn(samples)  # noqa: SLF001

    assert batch["state"].shape == (2, 32)
    assert batch["action_reference_state"].shape == (2, 32)
    assert batch["actions"].shape == (2, 50, 32)
    assert batch["action_dim_mask"].shape == (2, 32)


def test_multiple_human_modes_are_rejected():
    config = unitree_train_config.UnitreeExperimentDataConfig(robot_mode=2, human_modes=(3, 4))
    with pytest.raises(ValueError, match="at most one Human mode"):
        config.specs()


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robot1_abs", (1, 0, "absolute", "shared")),
        ("robot2_rel_hybrid", (2, 0, "relative", "hybrid")),
        ("human4_rel_per_step", (0, 4, "relative", "per_step")),
        ("mix_robot2_human6_rel_shared", (2, 6, "relative", "shared")),
    ],
)
def test_unitree_preset_parser(name, expected):
    preset = unitree_train_config.parse_unitree_preset(name)
    assert (preset.robot_mode, preset.human_mode, preset.action_representation, preset.relative_norm) == expected


@pytest.mark.parametrize(
    "name",
    [
        "robot1_abs_single",
        "robot1_gripper_abs_single",
        "robot2_abs_single",
        "robot2_rel_per_step_single",
        "robot2_gripper_rel_hybrid_single",
        "mix_robot2_human3_abs_progress_single",
        "mix_robot2_human6_gripper_eef_only_rel_shared_progress_single",
    ],
)
def test_single_suffix_supports_every_robot_training_family(name):
    preset = unitree_train_config.parse_unitree_preset(name)

    assert preset.robot_mode in (1, 2)
    assert preset.robot_camera_mode == "single"


@pytest.mark.parametrize(
    "name",
    [
        "robot1_abs",
        "robot2_rel_shared",
        "mix_robot2_human4_gripper_abs_progress",
    ],
)
def test_existing_robot_presets_remain_three_camera(name):
    assert unitree_train_config.parse_unitree_preset(name).robot_camera_mode == "three"


def test_single_suffix_is_rejected_without_a_robot_component():
    with pytest.raises(ValueError, match="requires a Robot component"):
        unitree_train_config.parse_unitree_preset("human3_abs_single")


@pytest.mark.parametrize(
    "name",
    [
        "robot1_gripper_abs",
        "robot2_gripper_abs",
        "robot2_gripper_rel_hybrid",
        "human4_gripper_rel_per_step",
        "mix_robot2_human6_gripper_rel_shared",
    ],
)
def test_gripper_presets_select_the_new_action_domain(name):
    assert unitree_train_config.parse_unitree_preset(name).action_domain == "gripper"


@pytest.mark.parametrize(
    ("name", "robot_domain", "representation", "norm", "progress"),
    [
        ("mix_robot2_human3_eef_only_abs", "brainco", "absolute", "shared", False),
        ("mix_robot2_human4_eef_only_rel_hybrid_progress", "brainco", "relative", "hybrid", True),
        ("mix_robot2_human5_gripper_eef_only_abs", "gripper", "absolute", "shared", False),
        (
            "mix_robot2_human6_gripper_eef_only_rel_shared_progress",
            "gripper",
            "relative",
            "shared",
            True,
        ),
    ],
)
def test_eef_only_human_presets_keep_the_robot_deployment_domain(name, robot_domain, representation, norm, progress):
    preset = unitree_train_config.parse_unitree_preset(name)

    assert preset.human_eef_only
    assert preset.action_domain == robot_domain
    assert preset.action_representation == representation
    assert preset.relative_norm == norm
    assert preset.task_progress_alignment == progress


def test_all_eef_only_mixture_presets_parse_for_every_human_mode_and_robot_domain():
    representations = ("abs", "rel_shared", "rel_per_step", "rel_hybrid")
    for human_mode in (3, 4, 5, 6):
        for robot_domain, domain_segment in (("brainco", ""), ("gripper", "gripper_")):
            for representation in representations:
                for progress in (False, True):
                    suffix = "_progress" if progress else ""
                    name = f"mix_robot2_human{human_mode}_{domain_segment}eef_only_{representation}{suffix}"
                    preset = unitree_train_config.parse_unitree_preset(name)

                    assert preset.human_eef_only
                    assert preset.action_domain == robot_domain
                    assert preset.task_progress_alignment == progress


def test_eef_only_remains_opt_in_for_every_legacy_mixture_preset():
    representations = ("abs", "rel_shared", "rel_per_step", "rel_hybrid")
    for human_mode in (3, 4, 5, 6):
        for domain_segment in ("", "gripper_"):
            for representation in representations:
                for progress in (False, True):
                    suffix = "_progress" if progress else ""
                    name = f"mix_robot2_human{human_mode}_{domain_segment}{representation}{suffix}"

                    assert not unitree_train_config.parse_unitree_preset(name).human_eef_only


def test_unitree_preset_parser_rejects_multiple_human_modes():
    with pytest.raises(ValueError, match="Unknown Unitree preset"):
        unitree_train_config.parse_unitree_preset("mix_robot2_human3_human4_rel_shared")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("robot1_abs", (1, 0, "absolute", "shared")),
        ("robot2_rel_per_step", (2, 0, "relative", "per_step")),
        ("human_abs", (0, 3, "absolute", "shared")),
        ("human_rel_hybrid", (0, 3, "relative", "hybrid")),
    ],
)
def test_unitree_norm_preset_selects_one_numeric_domain(name, expected):
    preset = unitree_train_config.parse_unitree_norm_preset(name)
    assert (preset.robot_mode, preset.human_mode, preset.action_representation, preset.relative_norm) == expected


@pytest.mark.parametrize(
    ("single_name", "three_name"),
    [
        ("robot1_abs_single", "robot1_abs"),
        ("robot1_gripper_abs_single", "robot1_gripper_abs"),
        ("robot2_abs_single", "robot2_abs"),
        ("robot2_rel_hybrid_single", "robot2_rel_hybrid"),
        ("robot2_gripper_rel_shared_single", "robot2_gripper_rel_shared"),
    ],
)
def test_single_camera_normalization_alias_keeps_the_numeric_domain(single_name, three_name):
    single = unitree_train_config.parse_unitree_norm_preset(single_name)
    three = unitree_train_config.parse_unitree_norm_preset(three_name)

    assert single.robot_camera_mode == "single"
    assert dataclasses.replace(single, robot_camera_mode="three") == three


def test_unitree_norm_preset_rejects_mixture():
    with pytest.raises(ValueError, match="exactly one Robot or Human domain"):
        unitree_train_config.parse_unitree_norm_preset("mix_robot2_human3_rel_shared")


def test_gripper_norm_preset_selects_one_numeric_domain():
    preset = unitree_train_config.parse_unitree_norm_preset("human_gripper_rel_hybrid")
    assert (preset.human_mode, preset.action_domain, preset.relative_norm) == (3, "gripper", "hybrid")


@pytest.mark.parametrize(
    "name",
    [
        "human_eef_only_abs",
        "human_eef_only_rel_shared",
        "human_eef_only_rel_per_step",
        "human_eef_only_rel_hybrid",
        "human_eef_only_abs_progress",
        "human_eef_only_robot_gripper_rel_shared_progress",
    ],
)
def test_eef_only_normalization_presets_select_native_human_eef18(name):
    preset = unitree_train_config.parse_unitree_norm_preset(name)

    assert preset.robot_mode == 0
    assert preset.human_mode == 3
    assert preset.human_eef_only


def test_joint16_gripper_norm_preset_is_absolute_robot_mode1():
    preset = unitree_train_config.parse_unitree_norm_preset("robot1_gripper_abs")
    assert (preset.robot_mode, preset.action_domain, preset.action_representation) == (1, "gripper", "absolute")


def test_progress_preset_is_opt_in_and_legacy_absolute_mix_stays_unchanged():
    progress = unitree_train_config.parse_unitree_preset("mix_robot2_human3_abs_progress")
    legacy = unitree_train_config.parse_unitree_preset("mix_robot2_human3_abs")

    assert progress.task_progress_alignment
    assert not legacy.task_progress_alignment
    assert (progress.robot_mode, progress.human_mode, progress.action_representation) == (2, 3, "absolute")
    assert unitree_train_config.parse_unitree_norm_preset("human_abs_progress").task_progress_alignment


@pytest.mark.parametrize(
    ("name", "domain", "representation", "norm"),
    [
        ("mix_robot2_human6_abs_progress", "brainco", "absolute", "shared"),
        ("mix_robot2_human5_rel_hybrid_progress", "brainco", "relative", "hybrid"),
        ("mix_robot2_human4_gripper_abs_progress", "gripper", "absolute", "shared"),
        ("mix_robot2_human3_gripper_rel_per_step_progress", "gripper", "relative", "per_step"),
    ],
)
def test_progress_suffix_supports_all_eef_mixture_families(name, domain, representation, norm):
    preset = unitree_train_config.parse_unitree_preset(name)

    assert preset.task_progress_alignment
    assert (preset.action_domain, preset.action_representation, preset.relative_norm) == (
        domain,
        representation,
        norm,
    )


@pytest.mark.parametrize(
    "name",
    [
        "human_abs_progress",
        "human_rel_hybrid_progress",
        "human_gripper_abs_progress",
        "human_gripper_rel_shared_progress",
    ],
)
def test_progress_normalization_presets_are_single_human_domains(name):
    preset = unitree_train_config.parse_unitree_norm_preset(name)

    assert preset.task_progress_alignment
    assert (preset.robot_mode, preset.human_mode) == (0, 3)


def test_task_progress_ratio_uses_median_complete_episode_durations(tmp_path):
    robot = tmp_path / "robot"
    human = tmp_path / "human"
    for dataset, lengths in ((robot, [101, 201]), (human, [51, 101])):
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text(json.dumps({"fps": 10}))
        (dataset / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"fold clothes."}\n')
        (dataset / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": index, "task_index": 0, "length": length}) + "\n"
                for index, length in enumerate(lengths)
            )
        )

    alignment = unitree_train_config.estimate_task_progress_alignment(robot, (0, 1), human, (0, 1))

    assert alignment["gamma_robot_over_human"] == pytest.approx(2.0)
    assert alignment["human_source_step_scale"] == pytest.approx(0.5)
    assert alignment["robot_episode_count"] == alignment["human_episode_count"] == 2
