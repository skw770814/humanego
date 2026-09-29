"""Self-contained checkpoint serving: no training dataset required at deployment."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import time

import numpy as np

from openpi.xrpipe_xinhai.config import ADAPTER, contract


@dataclasses.dataclass
class Args:
    checkpoint_dir: str
    port: int = 5000
    default_prompt: str | None = None
    host: str = "0.0.0.0"


class CheckedPolicy:
    def __init__(self, policy, manifest, default_prompt=None):
        self.policy, self.manifest, self.default_prompt = policy, manifest, default_prompt

    def infer(self, request):
        if request.get("adapter") != ADAPTER or request.get("contract_digest") != self.manifest["dataset_contract"]["digest"]:
            raise ValueError("Client coordinate adapter or training contract differs")
        state, rgb = np.asarray(request["state"]), np.asarray(request["rgb"])
        if state.shape != (19,) or not np.isfinite(state).all() or state[-1] not in (0, 1):
            raise ValueError("Expected finite canonical state[19] ending with actual binary gripper feedback")
        if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected original D405 RGB uint8[480,640,3]")
        prompt = request.get("task") or self.default_prompt
        if not prompt:
            raise ValueError("A policy task or server default prompt is required")
        start = time.perf_counter()
        result = self.policy.infer({"images": {"camera0": rgb}, "state": state, "prompt": prompt})
        actions = np.asarray(result["actions"])
        if actions.shape != (50, 10) or not np.isfinite(actions).all():
            raise ValueError("Invalid policy action output")
        return dict(actions=actions, observation_id=request["observation_id"], stamp_ns=request["stamp_ns"],
                    adapter=ADAPTER, policy_seconds=time.perf_counter() - start)


def build_policy(args, manifest):
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import policy_config, xrpipe_policy
    from openpi.training import config, checkpoints

    @dataclasses.dataclass(frozen=True)
    class InferenceData(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            return config.DataConfig(
                asset_id=manifest["asset_id"], use_quantile_norm=True, normalization_clip=5.,
                # Client already applied CanonicalizeRelationState. No action reference is needed on input.
                data_transforms=transforms.Group(inputs=[xrpipe_policy.XRPipeInputs(19, 10)],
                                                 outputs=[xrpipe_policy.XRPipeOutputs(10)]),
                model_transforms=config.ModelTransformFactory(track_modalities=True, mask_padded_action_dims=True)(model_config))

    train_config = config.TrainConfig(
        name="xrpipe_mode1_train", exp_name="inference",
        model=pi0_config.Pi0RTCConfig(pi05=True, discrete_state_input=True, action_dim=32, action_horizon=50),
        data=InferenceData(), policy_metadata={**manifest, "inference_adapter": ADAPTER})
    stats = checkpoints.load_norm_stats(Path(args.checkpoint_dir) / "assets", manifest["asset_id"])
    for key, size in (("state", 19), ("actions", 10)):
        if key not in stats:
            raise ValueError(f"Missing checkpoint normalization statistics: {key}")
        for field in ("mean", "std", "q01", "q99"):
            value = np.asarray(getattr(stats[key], field), dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key}.{field}: expected finite ({size},)")
    return policy_config.create_trained_policy(train_config, Path(args.checkpoint_dir),
                                               norm_stats=stats, default_prompt=args.default_prompt)


def main(args):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    manifest = contract(args.checkpoint_dir)
    if manifest.get("observation_only"):
        raise ValueError("Diagnostic contract cannot serve a policy; use the trained checkpoint manifest")
    asset_manifest = contract(Path(args.checkpoint_dir) / "assets" / manifest["asset_id"] / "runtime_manifest.json")
    if asset_manifest != manifest:
        raise ValueError("Root and normalization asset runtime manifests differ")
    start = time.perf_counter()
    policy = CheckedPolicy(build_policy(args, manifest), manifest, args.default_prompt)
    load_s = time.perf_counter() - start
    state = np.r_[np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0], 2), 0].astype(np.float32)
    policy.infer(dict(adapter=ADAPTER, contract_digest=manifest["dataset_contract"]["digest"],
                      state=state, rgb=np.zeros((480, 640, 3), np.uint8),
                      task=args.default_prompt or "warmup", observation_id="warmup", stamp_ns=0))
    logging.info("Policy ready: load %.2fs, total including warmup %.2fs", load_s, time.perf_counter() - start)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port,
                          metadata={**manifest, "inference_adapter": ADAPTER}).serve_forever()


if __name__ == "__main__":
    import tyro
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
