import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        if observation.action_pad_mask is None:
            return jnp.mean(chunked_loss)
        valid = jnp.logical_not(observation.action_pad_mask)
        return jnp.sum(chunked_loss * valid) / jnp.maximum(jnp.sum(valid), 1)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _reduce_chunk_loss(chunked_loss, observation: _model.Observation):
    if observation.action_pad_mask is None:
        return jnp.mean(chunked_loss)
    valid = jnp.logical_not(observation.action_pad_mask)
    return jnp.sum(chunked_loss * valid) / jnp.maximum(jnp.sum(valid), 1)


@at.typecheck
def eval_step(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()
    observation, actions = batch
    return _reduce_chunk_loss(model.compute_loss(rng, observation, actions, train=False), observation)


def _ablate_observation(observation: _model.Observation, modality: str) -> _model.Observation:
    """Build a diagnostic-only observation while preserving its JAX tree structure."""
    if modality == "image":
        return dataclasses.replace(
            observation,
            image_masks={key: jnp.zeros_like(value) for key, value in observation.image_masks.items()},
        )
    prompt_mask = observation.tokenized_prompt_mask
    if prompt_mask is None:
        return observation
    if modality == "text":
        span = observation.tokenized_text_mask
    elif modality == "state":
        span = observation.tokenized_state_mask
    else:
        raise ValueError(f"Unknown modality: {modality}")
    span = jnp.zeros_like(prompt_mask) if span is None else span
    return dataclasses.replace(observation, tokenized_prompt_mask=prompt_mask & ~span)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    peval_step = jax.jit(
        eval_step,
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    validation_loaders = {}
    validation_weights = {}
    if (config.validation_interval > 0 or config.modality_diagnostics_interval > 0) and config.validation_batches > 0:
        validation_loaders = _data_loader.create_validation_data_loaders(
            config,
            sharding=data_sharding,
            num_batches=config.validation_batches,
        )
        resolved_data_config = data_loader.data_config()
        if resolved_data_config.mixture_components:
            validation_weights = dict(
                zip(resolved_data_config.mixture_names, resolved_data_config.mixture_weights, strict=True)
            )
        else:
            validation_weights = dict.fromkeys(validation_loaders, 1.0)
        logging.info("Validation domains: %s", tuple(validation_loaders))
    if config.modality_diagnostics_interval > 0:
        wandb.run.summary["modality_diagnostic"] = (
            "controlled validation-loss ablation proxy; it is not a raw transformer attention weight"
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        completed_step = step + 1
        if validation_loaders and config.validation_interval > 0 and completed_step % config.validation_interval == 0:
            validation_metrics = {}
            weighted_loss = 0.0
            total_weight = 0.0
            for domain_index, (name, loader) in enumerate(validation_loaders.items()):
                losses = []
                for batch_index, validation_batch in enumerate(loader):
                    validation_rng = jax.random.fold_in(
                        train_rng, completed_step * 10_000 + domain_index * 100 + batch_index
                    )
                    with sharding.set_mesh(mesh):
                        validation_loss = peval_step(validation_rng, train_state, validation_batch)
                        jax.block_until_ready(validation_loss)
                        losses.append(validation_loss)
                domain_loss = float(np.mean(jax.device_get(jnp.stack(losses))))
                validation_metrics[f"val/loss/{name}"] = domain_loss
                weight = validation_weights.get(name, 1.0)
                weighted_loss += weight * domain_loss
                total_weight += weight
            validation_metrics["val/loss_weighted"] = weighted_loss / total_weight
            wandb.log(validation_metrics, step=completed_step)
            pbar.write(
                f"Step {completed_step}: "
                + ", ".join(f"val/{name}={validation_metrics[f'val/loss/{name}']:.4f}" for name in validation_loaders)
            )

        if (
            validation_loaders
            and config.modality_diagnostics_interval > 0
            and completed_step % config.modality_diagnostics_interval == 0
        ):
            modality_metrics = {}
            for domain_index, (name, loader) in enumerate(validation_loaders.items()):
                validation_batch = next(iter(loader))
                diagnostic_rng = jax.random.fold_in(train_rng, completed_step * 10_000 + 9_000 + domain_index)
                observation, actions = validation_batch
                with sharding.set_mesh(mesh):
                    baseline = peval_step(diagnostic_rng, train_state, validation_batch)
                    jax.block_until_ready(baseline)
                    ablated_losses = []
                    for modality in ("text", "state", "image"):
                        ablated_loss = peval_step(
                            diagnostic_rng,
                            train_state,
                            (_ablate_observation(observation, modality), actions),
                        )
                        jax.block_until_ready(ablated_loss)
                        ablated_losses.append(ablated_loss)
                    ablated = jnp.stack(ablated_losses)
                baseline = float(jax.device_get(baseline))
                deltas = np.asarray(jax.device_get(ablated)) - baseline
                positive = np.maximum(deltas, 0.0)
                shares = positive / max(float(np.sum(positive)), 1e-8)
                modality_metrics[f"val/modality_reliance/{name}/baseline"] = baseline
                for index, modality in enumerate(("text", "state", "image")):
                    modality_metrics[f"val/modality_reliance/{name}/{modality}_delta"] = float(deltas[index])
                    modality_metrics[f"val/modality_reliance/{name}/{modality}_share"] = float(shares[index])
            wandb.log(modality_metrics, step=completed_step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
