"""Offline tests; no RPC connection, checkpoints or model downloads required."""
from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from openpi.xrpipe_xinhai.config import ROOT, ObservationArgs, contract, pipeline_imports, prompts_for, read_json
from openpi.xrpipe_xinhai.geometry import S, from_vec9, pose7, relation_state, resample, targets, to_pose7, vec9
from openpi.xrpipe_xinhai.observation import GripperFilter, ObservationPipeline, RobotReader, decode_rgbd, save_snapshot
from openpi.xrpipe_xinhai.perception import ObjectTracker, ResidentPerception
from openpi.xrpipe_xinhai.visualization import visualize
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from robot_client_xrpipe_xinhai_ws import Args, execute_chunk
from serve_policy_xrpipe_xinhai import CheckedPolicy
from openpi.xrpipe_xinhai.config import ADAPTER

CONTRACT = ROOT / "configs/xrpipe_xinhai/observation_contract.json"


def action_chunk():
    actions = np.tile(np.r_[vec9(np.eye(4)), 0], (50, 1))
    actions[:, 0] = np.arange(1, 51) * .001
    actions[10:, 9] = 1
    return actions


def matrix(translation=(0, 0, 0), angles=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
    t[:3, 3] = translation
    return t


def training_math():
    # Execute the ACTUAL unchanged training transform definitions, not a duplicate.
    # Exclude dependency-heavy imports only: DataTransformFn is a typing protocol.
    path = ROOT / "src/openpi/policies/xrpipe_policy.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    module = types.ModuleType("_training_math_under_test")
    import sys
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, dataclasses=dataclasses, transforms=types.SimpleNamespace(DataTransformFn=object))
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class FakePerception:
    def __init__(self):
        self.calls = 0

    def process(self, rgb, stamp, **kwargs):
        self.calls += 1
        masks = np.zeros((2, 480, 640), bool)
        masks[0, 100:160, 160:270] = True
        masks[1, 290:380, 350:530] = True
        return dict(masks=masks, boxes=[[160, 100, 270, 160], [350, 290, 530, 380]],
                    scores=[.9, .8], score_kind="mock", timing={})


class Tests(unittest.TestCase):
    def test_openpi_folder_contains_runtime_sources_and_default_extrinsics(self):
        args = ObservationArgs(contract=str(CONTRACT))
        self.assertEqual(Path(args.extrinsics), ROOT / "extrinsics.json")
        self.assertIsNone(read_json(args.extrinsics)["T_B_C"])
        pipeline_imports()
        import xrrel
        import ego_relation
        import sam2
        bundled = ROOT / "third_party/xrpipe_source"
        for module in (xrrel, ego_relation, sam2):
            self.assertTrue(Path(module.__file__).is_relative_to(bundled), module.__file__)
        self.assertTrue(Path(args.dino_checkpoint, "model.safetensors").is_file())
        self.assertTrue(Path(args.sam2_checkpoint).is_file())

    def test_real_training_state_and_action_math(self):
        train = training_math()
        rng = np.random.default_rng(8)
        for _ in range(20):
            hand = matrix(rng.normal(size=3), rng.normal(size=3))
            objects = [matrix(rng.normal(size=3), rng.normal(size=3)) for _ in range(2)]
            raw, model = relation_state(hand, objects, True)
            expected = train.CanonicalizeRelationState(19)({"state": raw.copy()})["state"]
            np.testing.assert_allclose(model, expected, atol=2e-6)
            future = hand @ matrix((.01, .02, -.01), (.01, -.02, .03))
            converted = train.RelativeFingertipActions()({
                "action_reference": vec9(S @ hand @ S),
                "actions": np.tile(np.r_[vec9(S @ future @ S), 1], (50, 1))})["actions"]
            tem = matrix((0, 0, -.078), (.2, -.3, .4))
            poses, grips = targets(hand @ np.linalg.inv(tem), tem, converted, 20, 30, False)
            for pose in poses:
                np.testing.assert_allclose(pose7(pose) @ tem, future, atol=2e-6)
            np.testing.assert_array_equal(grips, 0)

    def test_five_keypoint_axes(self):
        pipeline_imports()
        from xrhand import gripper
        positions = np.zeros((26, 3))
        profile = read_json(ROOT / "configs/xrpipe_xinhai/right_gripper.json")
        for key in ("WRIST", "THUMB_BASE", "INDEX_BASE", "THUMB_TIP", "INDEX_TIP"):
            positions[getattr(gripper, key)] = profile["five_keypoints_E"][key.lower()]
        tem = np.asarray(profile["T_E_M"])
        np.testing.assert_allclose(gripper.midpoint_frame(positions), tem[:3, :3])
        np.testing.assert_allclose(gripper.midpoint(positions), tem[:3, 3])

    def test_aligned_profile_defaults_and_five_keypoints(self):
        from test_xrpipe_xinhai_observation import Args as ObservationTestArgs
        from openpi.xrpipe_xinhai.config import load_geometry
        pipeline_imports()
        from xrhand import gripper

        aligned_path = ROOT / "configs/xrpipe_xinhai/right_gripper_train_aligned.json"
        original_path = ROOT / "configs/xrpipe_xinhai/right_gripper.json"
        for cls in (ObservationArgs, Args, ObservationTestArgs):
            self.assertEqual(Path(cls(contract=str(CONTRACT)).tcp_geometry), aligned_path)
        args = ObservationArgs(contract=str(CONTRACT), extrinsics=None)
        aligned, new, _ = load_geometry(args)
        args.tcp_geometry = str(original_path)
        original, old, _ = load_geometry(args)
        q = np.diag([-1., 1., -1., 1.])
        np.testing.assert_allclose(new, old @ q, atol=1e-15)
        np.testing.assert_array_equal(new[:3, 3], old[:3, 3])
        np.testing.assert_array_equal(new[:3, 1], old[:3, 1])
        self.assertAlmostEqual(np.linalg.det(new[:3, :3]), 1.)
        self.assertEqual(aligned["five_keypoints_E"]["wrist"], original["five_keypoints_E"]["wrist"])
        for suffix in ("base", "tip"):
            self.assertEqual(aligned["five_keypoints_E"][f"thumb_{suffix}"],
                             original["five_keypoints_E"][f"index_{suffix}"])
            self.assertEqual(aligned["five_keypoints_E"][f"index_{suffix}"],
                             original["five_keypoints_E"][f"thumb_{suffix}"])
        positions = np.zeros((26, 3))
        for key in ("WRIST", "THUMB_BASE", "INDEX_BASE", "THUMB_TIP", "INDEX_TIP"):
            positions[getattr(gripper, key)] = aligned["five_keypoints_E"][key.lower()]
        np.testing.assert_allclose(gripper.midpoint_frame(positions), new[:3, :3], atol=1e-12)
        np.testing.assert_allclose(gripper.midpoint(positions), new[:3, 3], atol=1e-12)

    def test_profile_local_z_rotation_and_rollback(self):
        old = np.asarray(read_json(ROOT / "configs/xrpipe_xinhai/right_gripper.json")["T_E_M"])
        new = np.asarray(read_json(ROOT / "configs/xrpipe_xinhai/right_gripper_train_aligned.json")["T_E_M"])
        anchor_e = matrix((.3, -.1, .4), (.2, -.3, .4))
        anchor_m = anchor_e @ old
        delta = matrix(angles=(0, 0, .1))
        actions = np.tile(np.r_[vec9(delta), 0], (50, 1))
        old_targets, old_grips = targets(anchor_e, old, actions, 20, 30, False)
        new_targets, new_grips = targets(anchor_e, new, actions, 20, 30, False)
        # Compare physical rotations in the SAME original midpoint axes, not quaternion signs.
        for poses, angle in ((old_targets, .1), (new_targets, -.1)):
            physical_m = pose7(poses[0]) @ old
            relative = np.linalg.inv(anchor_m) @ physical_m
            np.testing.assert_allclose(Rotation.from_matrix(relative[:3, :3]).as_rotvec(),
                                       [0, 0, angle], atol=1e-12)
            np.testing.assert_allclose(physical_m[:3, 3], anchor_m[:3, 3], atol=1e-12)
        np.testing.assert_array_equal(new_grips, old_grips)
        rolled_back, _ = targets(anchor_e, old, actions, 20, 30, False)
        np.testing.assert_array_equal(rolled_back, old_targets)

    def test_both_profiles_shared_replay_state_targets_and_visualization(self):
        train = training_math()
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            extrinsics = dict(units="m", control_frame="torso_link4",
                              camera_frame="hdas/camera_wrist_right_color_optical_frame", T_B_C=np.eye(4).tolist())
            extra_path = temp / "extrinsics.json"
            extra_path.write_text(json.dumps(extrinsics))
            tcp = np.array([0, 0, .6, 0, 0, 0, 1.])
            for name in ("right_gripper", "right_gripper_train_aligned"):
                with self.subTest(profile=name):
                    args = ObservationArgs(contract=str(CONTRACT), extrinsics=str(extra_path),
                                           tcp_geometry=str(ROOT / f"configs/xrpipe_xinhai/{name}.json"))
                    pipeline = ObservationPipeline(args, FakePerception())
                    metadata = dict(stamp_ns=1_000_000_000, gripper_width=80., closed=False,
                                    tcp_before=tcp.tolist(), tcp_bracket_seconds=0.,
                                    camera=pipeline.camera, geometry=pipeline.geometry, extrinsics=extrinsics)
                    source = temp / f"{name}.npz"
                    np.savez_compressed(source, rgb=np.zeros((480, 640, 3), np.uint8),
                                        depth=np.full((480, 640), .5, np.float32), tcp=tcp,
                                        metadata=json.dumps(metadata))
                    obs = pipeline.capture_observation(replay=source)
                    hand = pose7(tcp) @ pipeline.tem
                    np.testing.assert_allclose(obs["T_C_M"], hand, atol=1e-12)
                    raw = np.r_[np.concatenate([vec9(np.linalg.inv(hand) @ o)
                                                for o in obs["T_C_O"]]), 0].astype(np.float32)
                    np.testing.assert_allclose(obs["raw_state"], raw, atol=1e-6)
                    canonical = train.CanonicalizeRelationState(19)({"state": raw.copy()})["state"]
                    np.testing.assert_allclose(obs["state"], canonical, atol=2e-6)
                    future = hand @ matrix((.01, -.02, .005), (.03, -.02, .04))
                    chunk = train.RelativeFingertipActions()({
                        "action_reference": vec9(S @ hand @ S),
                        "actions": np.tile(np.r_[vec9(S @ future @ S), 0], (50, 1))})["actions"]
                    poses, _ = targets(pose7(tcp), pipeline.tem, chunk, 20, 30, False)
                    np.testing.assert_allclose(pose7(poses[0]) @ pipeline.tem, future, atol=2e-6)
                    output = temp / name
                    visualize(obs, output, ("obj1", "obj2"))
                    with np.load(output / "snapshot.npz", allow_pickle=False) as z:
                        saved = json.loads(str(z["metadata"]))
                    self.assertEqual(saved["geometry"], pipeline.geometry)
                    report = read_json(output / "report.json")
                    self.assertEqual(report["geometry"], pipeline.geometry)
                    self.assertIn("fingertip_midpoint", report["projections"])
                    np.testing.assert_allclose(report["training_state"], obs["state"])
                    # A snapshot recorded under one profile must not silently replay under the other.
                    other = "right_gripper" if name == "right_gripper_train_aligned" else "right_gripper_train_aligned"
                    wrong = dataclasses.replace(args, tcp_geometry=str(ROOT / f"configs/xrpipe_xinhai/{other}.json"))
                    with self.assertRaisesRegex(ValueError, "Replay calibration mismatch"):
                        ObservationPipeline(wrong, FakePerception()).capture_observation(replay=source)

    def test_resampling_and_anchor(self):
        actions = action_chunk()
        delta, grips = resample(actions, 20, 30, False)
        np.testing.assert_allclose(delta[:, 0, 3], actions[:20, 0])
        self.assertEqual(len(grips), 20)
        faster, fast_grip = resample(actions, 20, 60, True)
        np.testing.assert_allclose(faster[:, 0, 3], np.arange(1, 21) * .0005)
        self.assertTrue(fast_grip[0])  # current gripper is held before first training timestamp
        self.assertFalse(fast_grip[1])
        slow, _ = resample(actions, 20, 15, False)
        np.testing.assert_allclose(slow[:, 0, 3], np.arange(1, 21) * .002)
        with self.assertRaises(ValueError):
            resample(actions, 30, 15, False)
        with self.assertRaises(ValueError):
            resample(actions, 51, 30, False)
        actions[-1, 3:9] = 0
        with self.assertRaises(ValueError):
            resample(actions, 1, 30, False)  # reject even invalid unexecuted chunk tail

    def test_slerp(self):
        actions = action_chunk()
        for i in range(50):
            actions[i, :9] = vec9(matrix(angles=(0, 0, (i+1)*.01)))
        delta, _ = resample(actions, 20, 60, False)
        np.testing.assert_allclose(Rotation.from_matrix(delta[:, :3, :3]).as_rotvec()[:, 2],
                                   np.arange(1, 21)*.005, atol=1e-8)

    def test_depth_stride_endianness_rgb(self):
        camera = dict(width=3, height=2, camera_frame="optical", depth_scale=.001)
        bgr = np.zeros((2, 3, 3), np.uint8)
        bgr[..., 2] = 200
        jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
        for endian in (False, True):
            raw = np.array([[1000, 0, 2000, 1234], [3000, 4000, 5000, 1234]], dtype=">u2" if endian else "<u2")
            packet = dict(frame_id="/optical", stamp=dict(sec=1, nanosec=2), data=jpeg, format="bgr8; jpeg compressed bgr8",
                          depth=dict(stamp=dict(sec=1, nanosec=2), encoding="16UC1", height=2, width=3,
                                     step=8, is_bigendian=endian, data=raw.tobytes()))
            rgb, depth, stamp = decode_rgbd(packet, camera)
            self.assertGreater(rgb[0, 0, 0], 190)
            self.assertTrue(np.isnan(depth[0, 1]))
            self.assertAlmostEqual(float(depth[1, 2]), 5)
            self.assertEqual(stamp, 1000000002)
            packet["depth"]["stamp"]["nanosec"] = 3
            with self.assertRaises(ValueError):
                decode_rgbd(packet, camera)

    def test_contract_prompts_and_bad_geometry(self):
        manifest = contract(CONTRACT)
        args = ObservationArgs(contract=str(CONTRACT))
        self.assertEqual(prompts_for(args, manifest), ("green type", "black plate"))
        args.object_prompts = ("bolt",)
        with self.assertRaises(ValueError):
            prompts_for(args, manifest)
        manifest["state_dim"] = 20
        from openpi.xrpipe_xinhai.config import validate_contract, load_geometry
        with self.assertRaises(ValueError):
            validate_contract(manifest)
        args.extrinsics = str(ROOT / "configs/xrpipe_xinhai/extrinsics.template.json")
        with self.assertRaises(ValueError):
            load_geometry(args)

    def test_gripper_filter_causal(self):
        filt = GripperFilter()
        for t in np.arange(0, .21, .02):
            filt.update(80, t)
        self.assertFalse(filt.closed)
        for t in np.arange(.22, .71, .02):
            filt.update(0, t)
        self.assertTrue(filt.closed)
        self.assertTrue(filt.update(80, .72))  # no premature backfill
        with self.assertRaises(ValueError):
            filt.update(float("nan"), .74)

    def test_live_and_replay_identical_shared_path_and_visualization(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            extrinsics = dict(units="m", control_frame="torso_link4",
                              camera_frame="hdas/camera_wrist_right_color_optical_frame", T_B_C=np.eye(4).tolist())
            extra_path = temp / "extrinsics.json"
            extra_path.write_text(json.dumps(extrinsics))
            args = ObservationArgs(contract=str(CONTRACT), extrinsics=str(extra_path))
            camera = read_json(args.camera_config)
            rgb = np.zeros((480, 640, 3), np.uint8)
            rgb[..., 1] = 120
            depth = np.full((480, 640), 500, dtype="<u2")
            stamp = time.time_ns()
            stamp_dict = dict(sec=stamp//10**9, nanosec=stamp%10**9)
            packet = dict(stamp=stamp_dict, frame_id=camera["camera_frame"], format="bgr8; jpeg compressed bgr8",
                          data=cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes(),
                          depth=dict(stamp=stamp_dict, encoding="16UC1", height=480, width=640, step=1280,
                                     is_bigendian=False, data=depth.tobytes()))
            # No move method at all: the shared observation path cannot issue control.
            def read_packet():
                stamp = time.time_ns()
                packet["stamp"] = dict(sec=stamp//10**9, nanosec=stamp%10**9)
                packet["depth"]["stamp"] = packet["stamp"]
                return packet
            reader = types.SimpleNamespace(read_tcp=lambda: np.array([0, 0, .6, 0, 0, 0, 1.]),
                                           read_gripper=lambda: 80., read_rgbd=read_packet)
            first = ObservationPipeline(args, FakePerception())
            obs = first.capture_observation(reader)
            save_snapshot(obs, temp / "live")
            replay_pipeline = ObservationPipeline(args, FakePerception())
            replay = replay_pipeline.capture_observation(replay=temp / "live/snapshot.npz")
            np.testing.assert_allclose(obs["state"], replay["state"], atol=1e-6)
            visualize(replay, temp / "vis", ("obj1", "obj2"))
            self.assertTrue((temp / "vis/geometry.png").is_file())
            self.assertTrue((temp / "vis/camera_scene_3d.png").is_file())
            with self.assertRaises(ValueError):
                replay_pipeline.capture_observation(replay=temp / "live/snapshot.npz")
            self.assertEqual(replay_pipeline.perception.calls, 1)
            self.assertFalse(hasattr(RobotReader, "move"))

    def test_latch_priority_occlusion_and_release(self):
        rng = np.random.default_rng(6)
        clouds = [rng.normal(scale=[.006, .004, .002], size=(300, 3)) + [0, 0, .5],
                  rng.normal(scale=[.006, .004, .002], size=(300, 3)) + [.01, 0, .5]]
        tracker = ObjectTracker()
        hand = matrix((0, 0, .5))
        initial, _ = tracker.update(clouds, [np.zeros(2)]*2, hand, True, 1.)
        self.assertEqual(tracker.lock[0], 0)
        moved = matrix((.02, 0, .5))
        out, report = tracker.update([np.empty((0, 3)), clouds[1]], [np.zeros(2)]*2, moved, True, 1.1)
        np.testing.assert_allclose(out[0], moved @ np.linalg.inv(hand) @ initial[0])
        self.assertTrue(report[0]["latched"])
        self.assertFalse(report[1]["latched"])
        tracker.feedback(moved, False, 1.2)
        self.assertIsNone(tracker.lock)

    def test_execution_prefix_no_left_ik_and_timeout(self):
        class Clock:
            value = 10.
            def __call__(self): return self.value
            def sleep(self, seconds): self.value += seconds
        with tempfile.TemporaryDirectory() as temp:
            args = Args(contract=str(CONTRACT), output_dir=temp, dry_run=False)
            clock = Clock()
            commands = []
            robot = types.SimpleNamespace(move=lambda p, g: commands.append((p, g)))
            pipeline = types.SimpleNamespace(poll_feedback=lambda r: None)
            obs = dict(received_monotonic=10., metadata=dict(observation_id="test"))
            poses, grips = targets(np.eye(4), np.eye(4), action_chunk(), 20, 30, False)
            result = execute_chunk(robot, pipeline, obs, poses, grips, args, clock, clock.sleep)
            self.assertEqual(len(commands), 20)
            self.assertEqual(len(result), 20)
            self.assertEqual(commands[0][1], 80)
            self.assertEqual(commands[-1][1], 0)
            commands.clear()
            clock.value = 10.
            def slow(_): clock.value += .2
            pipeline.poll_feedback = slow
            with self.assertRaises(TimeoutError):
                execute_chunk(robot, pipeline, obs, poses, grips, args, clock, clock.sleep)
            self.assertFalse(commands)

    def test_policy_no_second_canonicalization(self):
        manifest = contract(CONTRACT)
        received = []
        policy = types.SimpleNamespace(infer=lambda x: received.append(x) or {"actions": action_chunk()})
        checked = CheckedPolicy(policy, manifest, "default task")
        state = np.r_[vec9(np.eye(4)), vec9(np.eye(4)), 1].astype(np.float32)
        request = dict(adapter=ADAPTER, contract_digest=manifest["dataset_contract"]["digest"],
                       rgb=np.zeros((480, 640, 3), np.uint8), state=state, observation_id="1", stamp_ns=1)
        result = checked.infer(request)
        np.testing.assert_array_equal(received[0]["state"], state)
        self.assertEqual(result["observation_id"], "1")
        self.assertEqual(received[0]["prompt"], "default task")


if __name__ == "__main__":
    unittest.main()

