from collections.abc import Iterator, Sequence
import logging
import math
import multiprocessing
import os
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch

# LeRobot 0.3 moved dataset APIs out of ``lerobot.common``. Prefer the
# repository-pinned legacy path, while accepting the equivalent 0.3 API.
try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError as exc:
    if exc.name not in {
        "lerobot.common",
        "lerobot.common.datasets",
        "lerobot.common.datasets.lerobot_dataset",
    }:
        raise
    import lerobot.datasets.lerobot_dataset as lerobot_dataset

import openpi.models.model as _model
from openpi.training.action_chunk_resampling import required_source_horizon
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class EpisodeFrameSubset(Dataset[T_co]):
    """Select complete episodes without using LeRobot's broken sparse-episode indexing.

    LeRobot v2.1 builds compact episode bounds when ``episodes=`` is supplied, but
    later indexes those bounds with the original episode id.  Non-contiguous
    holdouts therefore fail at sample time.  This wrapper keeps the underlying
    dataset complete and maps a dense local frame index to the selected global
    episode ranges.
    """

    def __init__(self, dataset: Dataset[T_co], episode_ranges: Sequence[tuple[int, int]]):
        if not episode_ranges:
            raise ValueError("An episode subset must contain at least one episode")
        self._dataset = dataset
        self._starts = np.asarray([start for start, _ in episode_ranges], dtype=np.int64)
        lengths = np.asarray([end - start for start, end in episode_ranges], dtype=np.int64)
        if np.any(self._starts < 0) or np.any(lengths <= 0):
            raise ValueError(f"Invalid episode frame ranges: {episode_ranges}")
        self._ends = np.cumsum(lengths)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        value = index.__index__()
        if value < 0:
            value += len(self)
        if value < 0 or value >= len(self):
            raise IndexError(value)
        range_index = int(np.searchsorted(self._ends, value, side="right"))
        local_start = 0 if range_index == 0 else int(self._ends[range_index - 1])
        global_index = int(self._starts[range_index] + value - local_start)
        return self._dataset[global_index]

    def __len__(self) -> int:
        return int(self._ends[-1])


class WeightedMixtureDataset(Dataset[T_co]):
    """Concatenate transformed components while exposing their boundaries to a batch sampler."""

    def __init__(self, datasets: Sequence[Dataset[T_co]], weights: Sequence[float], names: Sequence[str]):
        if len(datasets) < 2:
            raise ValueError("A mixture requires at least two datasets")
        if len(weights) != len(datasets) or len(names) != len(datasets):
            raise ValueError("Mixture datasets, weights, and names must have equal lengths")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("Mixture components must be non-empty")
        if any(not np.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError(f"Mixture weights must be finite and positive, got {weights}")
        self.datasets = tuple(datasets)
        self.weights = np.asarray(weights, dtype=np.float64) / np.sum(weights)
        self.names = tuple(names)
        self.lengths = tuple(len(dataset) for dataset in datasets)
        self.offsets = tuple(np.cumsum((0, *self.lengths[:-1])).tolist())

    def __getitem__(self, index: SupportsIndex) -> T_co:
        value = index.__index__()
        if value < 0 or value >= len(self):
            raise IndexError(value)
        component_index = int(np.searchsorted(self.offsets, value, side="right") - 1)
        return self.datasets[component_index][value - self.offsets[component_index]]

    def __len__(self) -> int:
        return sum(self.lengths)


def mixture_batch_quotas(batch_size: int, weights: np.ndarray) -> np.ndarray:
    if batch_size < len(weights):
        raise ValueError(f"Batch size {batch_size} must be at least the number of mixture components {len(weights)}")
    exact = batch_size * weights
    quotas = np.maximum(1, np.floor(exact).astype(np.int64))
    while int(quotas.sum()) < batch_size:
        residual = exact - quotas
        quotas[int(np.argmax(residual))] += 1
    while int(quotas.sum()) > batch_size:
        candidates = np.where(quotas > 1, quotas - exact, -np.inf)
        index = int(np.argmax(candidates))
        if not np.isfinite(candidates[index]):
            raise ValueError("Cannot assign at least one sample to each mixture component")
        quotas[index] -= 1
    return quotas


class WeightedMixtureBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Emit fixed-composition batches, including deterministic DDP sharding."""

    def __init__(
        self,
        dataset: WeightedMixtureDataset,
        batch_size: int,
        *,
        seed: int,
        shuffle: bool,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.dataset = dataset
        self.quotas = mixture_batch_quotas(batch_size, dataset.weights)
        self.seed = seed
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        local_lengths = [len(range(rank, length, world_size)) for length in dataset.lengths]
        if any(length == 0 for length in local_lengths):
            raise ValueError("Every DDP rank must receive at least one sample from every mixture component")
        self.num_batches = max(
            math.ceil(length / quota) for length, quota in zip(local_lengths, self.quotas, strict=True)
        )

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + self.rank)
        pools = [np.arange(self.rank, length, self.world_size) for length in self.dataset.lengths]
        if self.shuffle:
            for pool in pools:
                rng.shuffle(pool)
        cursors = [0] * len(pools)

        for _ in range(self.num_batches):
            batch = []
            for component_index, (pool, quota) in enumerate(zip(pools, self.quotas, strict=True)):
                selected = []
                while len(selected) < quota:
                    available = len(pool) - cursors[component_index]
                    take = min(int(quota) - len(selected), available)
                    if take:
                        selected.extend(pool[cursors[component_index] : cursors[component_index] + take].tolist())
                        cursors[component_index] += take
                    if cursors[component_index] == len(pool):
                        cursors[component_index] = 0
                        if self.shuffle:
                            rng.shuffle(pool)
                batch.extend(self.dataset.offsets[component_index] + index for index in selected)
            if self.shuffle:
                rng.shuffle(batch)
            yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return self.num_batches


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    split: Literal["train", "validation"] = "train",
    skip_norm_stats: bool = False,
) -> Dataset:
    """Create a dataset for training."""
    if data_config.mixture_components:
        components = [
            transform_dataset(
                create_torch_dataset(
                    component,
                    action_horizon,
                    model_config,
                    split=split,
                    skip_norm_stats=skip_norm_stats,
                ),
                component,
                skip_norm_stats=skip_norm_stats,
            )
            for component in data_config.mixture_components
        ]
        weights = data_config.mixture_weights or tuple(1.0 for _ in components)
        names = data_config.mixture_names or tuple(
            component.asset_id or component.repo_id or f"component_{index}"
            for index, component in enumerate(data_config.mixture_components)
        )
        return WeightedMixtureDataset(components, weights, names)

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    source_horizon = (
        action_horizon
        if data_config.action_source_step_scale is None
        else required_source_horizon(action_horizon, data_config.action_source_step_scale)
    )
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(source_horizon)] for key in data_config.action_sequence_keys
        },
    )

    selected_episodes = data_config.train_episodes if split == "train" else data_config.validation_episodes
    if split == "validation" and selected_episodes is None:
        raise ValueError(f"No validation episodes configured for {data_config.asset_id or data_config.repo_id}")
    if selected_episodes is not None:
        unique_episodes = tuple(dict.fromkeys(int(index) for index in selected_episodes))
        total_episodes = dataset_meta.total_episodes
        invalid = [index for index in unique_episodes if index < 0 or index >= total_episodes]
        if invalid:
            raise ValueError(f"Episode indices out of range [0, {total_episodes}): {invalid[:10]}")
        episode_ranges = [
            (
                int(dataset.episode_data_index["from"][index]),
                int(dataset.episode_data_index["to"][index]),
            )
            for index in unique_episodes
        ]
        dataset = EpisodeFrameSubset(dataset, episode_ranges)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not data_config.mixture_components and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                clip_range=data_config.normalization_clip,
            ),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(
                norm_stats,
                use_quantiles=data_config.use_quantile_norm,
                clip_range=data_config.normalization_clip,
            ),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_validation_data_loaders(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    num_batches: int | None = None,
    framework: Literal["jax", "pytorch"] = "jax",
) -> dict[str, DataLoader[tuple[_model.Observation, _model.Actions]]]:
    """Create one deterministic validation loader per numeric/domain component."""
    resolved = config.data.create(config.assets_dirs, config.model)
    components = tuple(resolved.mixture_components) or (resolved,)
    names = tuple(resolved.mixture_names) or tuple(
        str(component.runtime_manifest.get("name"))
        if component.runtime_manifest and component.runtime_manifest.get("name")
        else component.asset_id or f"component_{index}"
        for index, component in enumerate(components)
    )
    loaders = {}
    for name, component in zip(names, components, strict=True):
        if component.validation_episodes is None:
            logging.warning("Skipping validation for %s: no validation episodes configured", name)
            continue
        loaders[name] = create_torch_data_loader(
            component,
            model_config=config.model,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=False,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=config.seed,
            framework=framework,
            split="validation",
        )
    return loaders


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    split: Literal["train", "validation"] = "train",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        split=split,
        skip_norm_stats=skip_norm_stats,
    )
    if not data_config.mixture_components:
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    rank = 0
    world_size = 1
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            if not isinstance(dataset, WeightedMixtureDataset):
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=shuffle,
                    drop_last=True,
                )
            local_batch_size = batch_size // world_size
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    batch_sampler = None
    if isinstance(dataset, WeightedMixtureDataset):
        batch_sampler = WeightedMixtureBatchSampler(
            dataset,
            local_batch_size,
            seed=seed,
            shuffle=shuffle,
            rank=rank,
            world_size=world_size,
        )
        logging.info(
            "Mixture batch quotas: %s for %s",
            batch_sampler.quotas.tolist(),
            dataset.names,
        )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler[list[int]] | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        loader_kwargs = {
            "dataset": typing.cast(torch.utils.data.Dataset, dataset),
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "collate_fn": _collate_fn,
            "worker_init_fn": _worker_init_fn,
            "generator": generator,
        }
        if batch_sampler is not None:
            self._data_loader = torch.utils.data.DataLoader(batch_sampler=batch_sampler, **loader_kwargs)
        else:
            self._data_loader = torch.utils.data.DataLoader(
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
                sampler=sampler,
                drop_last=True,
                **loader_kwargs,
            )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
