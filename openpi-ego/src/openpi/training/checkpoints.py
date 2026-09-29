from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import enum
import json
import logging
import os
import pathlib
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        save_data_assets(directory, data_loader.data_config())

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def save_data_assets(directory: epath.Path | pathlib.Path | str, data_config: _config.DataConfig) -> None:
    """Save normalization assets and runtime metadata for a data config.

    The legacy single-dataset layout remains ``assets/<asset_id>/norm_stats.json``.
    For a mixture, every leaf component is written to its own ``asset_id`` directory.
    Mixtures also receive an aggregate ``assets/runtime_manifest.json`` and a small
    component-local manifest, so evaluation can select and validate the intended
    runtime representation without depending on the training config.
    """
    directory = epath.Path(directory)
    components = _leaf_data_configs(data_config)

    serialized_stats: dict[str, str] = {}
    for component in components:
        if component.norm_stats is None or component.asset_id is None:
            continue
        serialized = _normalize.serialize_json(component.norm_stats)
        if component.asset_id in serialized_stats:
            previous = serialized_stats[component.asset_id]
            if previous != serialized:
                raise ValueError(
                    f"Mixture components share asset_id {component.asset_id!r} but have different norm stats"
                )
            continue
        serialized_stats[component.asset_id] = serialized
        _normalize.save(directory / component.asset_id, component.norm_stats)

    if data_config.mixture_components:
        entries = _component_manifest_entries(data_config, components)
        aggregate = dict(data_config.runtime_manifest or {})
        aggregate.setdefault("schema_version", 1)
        aggregate.setdefault("components", entries)
        _write_runtime_manifest(directory / "runtime_manifest.json", aggregate)

        for component, entry in zip(components, entries, strict=True):
            if component.asset_id is not None:
                component_manifest = {key: value for key, value in entry.items() if key != "runtime_manifest"}
                component_manifest.update(component.runtime_manifest or {})
                _write_runtime_manifest(directory / component.asset_id / "runtime_manifest.json", component_manifest)
    elif data_config.runtime_manifest is not None:
        # A manifest is opt-in for a legacy single dataset, preserving the exact
        # old behavior for configs that do not define one.
        _write_runtime_manifest(directory / "runtime_manifest.json", data_config.runtime_manifest)
        if data_config.asset_id is not None:
            _write_runtime_manifest(
                directory / data_config.asset_id / "runtime_manifest.json", data_config.runtime_manifest
            )


def _leaf_data_configs(data_config: _config.DataConfig) -> tuple[_config.DataConfig, ...]:
    if not data_config.mixture_components:
        return (data_config,)
    return tuple(leaf for component in data_config.mixture_components for leaf in _leaf_data_configs(component))


def _component_manifest_entries(
    data_config: _config.DataConfig,
    components: tuple[_config.DataConfig, ...],
) -> list[dict]:
    names = tuple(data_config.mixture_names)
    if names and len(names) != len(components):
        # Nested mixtures do not currently have a flattened name contract. Fall
        # back to stable asset/repository identifiers instead of mislabelling.
        names = ()

    entries = []
    for index, component in enumerate(components):
        entry = {
            "name": names[index] if names else component.asset_id or component.repo_id or f"component_{index}",
            "asset_id": component.asset_id,
            "repo_id": component.repo_id,
        }
        if component.runtime_manifest is not None:
            entry["runtime_manifest"] = component.runtime_manifest
        entries.append(entry)
    return entries


def _write_runtime_manifest(path: epath.Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
