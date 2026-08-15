"""
lerobot.py

Lightweight PyTorch `IterableDataset` for the LeRobot v3 dataset format (frame-level parquet + mp4 videos).

Why this exists:
    - The native MemoryVLA / CogACT data pipeline reads RLDS/TFDS only (`tfds.builder(name, data_dir)`).
    - WA01 is stored in LeRobot v3 format: `meta/episodes/` per-episode metadata parquet, `data/` per-episode
      frame parquet (incl. the 13-dim `action` column) and `videos/<camera>/...mp4` videos.
    - This adapter reads those files directly and emits raw samples that are 100% compatible with the existing
      `RLDSBatchTransform` + `PaddedCollatorForActionPrediction` + FSDP training loop, so the training
      framework needs *zero* invasive changes.

Group semantics:
    - Mirrors `GroupRLDSDataset`: each episode is cut into contiguous groups of `group_size` frames (frame order
      preserved within a group). `group_size` MUST equal `per_device_batch_size` so that each collated batch is
      exactly one group and the `CogMemBank` memory chain stays correct.
    - `action_dim` is inferred from the actual `action` column (e.g. 13 for WA01); no hardcoding here.
"""

import glob
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

import av
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset

from prismatic.overwatch import initialize_overwatch
from vla.datasets.datasets import RLDSBatchTransform

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DecodeTimeoutError(RuntimeError):
    """Raised when a video decode does not finish within `decode_timeout` seconds.

    Root cause this guards against: a pathological / corrupt AV1 stream (libdav1d) can stall the
    main-process decode forever *without raising an exception*, so a plain try/except can never
    recover. See `run_with_timeout` for the abandonment strategy.
    """


_T = TypeVar("_T")


def run_with_timeout(fn: Callable[..., _T], timeout_seconds: float, *args: Any, **kwargs: Any) -> _T:
    """Run `fn` in a fresh daemon thread and wait up to `timeout_seconds`.

    A truly stuck C call (e.g. libdav1d decoding a bad AV1 frame) cannot be interrupted from the
    main thread -- Python only services signal handlers after a blocking C call returns -- so we
    *abandon* the worker instead: `join(timeout)` returns, we raise `DecodeTimeoutError`, and the
    caller's retry/skip path takes over. The abandoned daemon thread is left to finish (or stay
    stuck) on its own core; this only leaks a thread for genuinely pathological episodes, which is
    exactly the case we want to survive.
    """
    holder: Dict[str, Any] = {"result": None, "error": None}

    def _worker() -> None:
        try:
            holder["result"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 -- re-raise *everything* (incl. KeyboardInterrupt) on main thread
            holder["error"] = e

    worker = threading.Thread(target=_worker, name="video-decode", daemon=True)
    worker.start()
    worker.join(timeout=timeout_seconds)
    if worker.is_alive():
        raise DecodeTimeoutError(
            f"video decode exceeded {timeout_seconds:.0f}s and was abandoned; "
            f"a background daemon thread is still draining it"
        )
    if holder["error"] is not None:
        raise holder["error"]  # type: ignore[misc]
    return holder["result"]  # type: ignore[return-value]


class LeRobotDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        dataset_name: str = "wa01",
        batch_transform: Optional[RLDSBatchTransform] = None,
        image_key: str = "observation.images.robot0_agentview_left",
        future_action_window_size: int = 15,
        group_size: int = 8,
        train: bool = True,
        seed: int = 0,
        decode_timeout: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.data_root_dir = Path(data_root_dir)
        self.dataset_name = dataset_name
        self.batch_transform = batch_transform
        self.image_key = image_key
        self.future_action_window_size = future_action_window_size
        self.action_window_size = future_action_window_size + 1  # == 16 for the default
        self.group_size = group_size
        self.train = train
        self.seed = seed
        # Hard per-video decode timeout (seconds). If a decode exceeds this, the episode is
        # abandoned (thread) and skipped via the existing retry/skip path instead of hanging the
        # whole run until the NCCL watchdog kills it. `None` => 120s default.
        self.decode_timeout = 120 if decode_timeout is None else decode_timeout
        if self.decode_timeout <= 0:
            raise ValueError(f"decode_timeout must be > 0, got {self.decode_timeout}")

        # Rank / world size sharding (dataset is built after `torch.distributed` is initialized)
        import torch.distributed as dist

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

        self.global_stats = self._load_global_stats()
        self.episodes = self._load_episodes()
        # Shard episodes across ranks without breaking frame order inside an episode.
        self.episodes = [ep for i, ep in enumerate(self.episodes) if i % self.world_size == self.rank]

    # === Metadata ===
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Read `meta/episodes/chunk-*/file-*.parquet` (one episode per file) and resolve data/video paths."""
        needed_cols = [
            "episode_index",
            "length",
            "tasks",
            "data/chunk_index",
            "data/file_index",
            f"videos/{self.image_key}/chunk_index",
            f"videos/{self.image_key}/file_index",
        ]
        episodes: List[Dict[str, Any]] = []
        meta_glob = sorted(glob.glob(str(self.data_root_dir / "meta" / "episodes" / "chunk-*" / "file-*.parquet")))
        if len(meta_glob) == 0:
            raise FileNotFoundError(f"No meta/episodes parquet files found under {self.data_root_dir}")
        for meta_path in meta_glob:
            t = pq.read_table(meta_path, columns=needed_cols)
            assert t.num_rows == 1, f"Expected one episode per meta file, got {t.num_rows}: {meta_path}"
            row = {c: t.column(c).to_pylist()[0] for c in needed_cols}

            data_chunk, data_file = row["data/chunk_index"], row["data/file_index"]
            video_chunk = row[f"videos/{self.image_key}/chunk_index"]
            video_file = row[f"videos/{self.image_key}/file_index"]

            episodes.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "length": int(row["length"]),
                    "instruction": row["tasks"][0],
                    "data_path": self.data_root_dir / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet",
                    "video_path": (
                        self.data_root_dir
                        / "videos"
                        / self.image_key
                        / f"chunk-{video_chunk:03d}"
                        / f"file-{video_file:03d}.mp4"
                    ),
                }
            )
        return episodes

    def _load_global_stats(self) -> Dict[str, Any]:
        """Read `meta/stats.json` `action` field and precompute the BOUNDS_Q99 normalization stats + mask."""
        stats_path = self.data_root_dir / "meta" / "stats.json"
        with open(stats_path, "r") as f:
            action_stats = json.load(f)["action"]

        q01 = np.asarray(action_stats["q01"], dtype=np.float32)
        q99 = np.asarray(action_stats["q99"], dtype=np.float32)
        return {
            "mean": np.asarray(action_stats["mean"], dtype=np.float32),
            "std": np.asarray(action_stats["std"], dtype=np.float32),
            "min": np.asarray(action_stats["min"], dtype=np.float32),
            "max": np.asarray(action_stats["max"], dtype=np.float32),
            "q01": q01,
            "q99": q99,
            "mask": q01 != q99,
        }

    # === Normalization (must be the inverse of `predict_action` de-normalization) ===
    def _normalize_actions(self, acts: np.ndarray) -> np.ndarray:
        """BOUNDS_Q99 normalization to [-1, 1]; constant dimensions are zeroed out (matches dlimp)."""
        q01, q99 = self.global_stats["q01"], self.global_stats["q99"]
        mask = self.global_stats["mask"]
        norm = np.clip(2.0 * (acts - q01) / (q99 - q01 + 1e-8) - 1.0, -1.0, 1.0)
        return np.where(mask, norm, 0.0)

    # === Dataset statistics (for `save_dataset_statistics` at inference time) ===
    @property
    def dataset_statistics(self) -> Dict[str, Dict[str, Any]]:
        stats = self.global_stats
        return {
            self.dataset_name: {
                "action": {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "q01": stats["q01"],
                    "q99": stats["q99"],
                    "mask": stats["mask"],
                },
                "num_transitions": sum(ep["length"] for ep in self.episodes),
                "num_trajectories": len(self.episodes),
            }
        }

    def __len__(self) -> int:
        # Count only full groups; the tail of each episode (that does not fill a group) is dropped.
        return int(sum((ep["length"] // self.group_size) * self.group_size for ep in self.episodes))

    # === Data loading ===
    @staticmethod
    def _read_actions(data_path: Path) -> np.ndarray:
        t = pq.read_table(data_path, columns=["action"])
        actions = t.column("action").to_numpy(zero_copy_only=False)
        # pyarrow returns an object array of lists for fixed_size_list; normalize to (L, A)
        if actions.dtype == object:
            actions = np.stack(actions.tolist()).astype(np.float32)
        else:
            actions = actions.reshape(-1, actions.shape[-1]).astype(np.float32)
        return actions

    @staticmethod
    def _decode_video(video_path: Path) -> np.ndarray:
        """Decode an entire video to (L, H, W, 3) uint8 using pyav (handles AV1/HEVC/H.264)."""
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        finally:
            container.close()
        if len(frames) == 0:
            raise RuntimeError(f"Decoded zero frames from {video_path}")
        return np.stack(frames).astype(np.uint8)

    # === Iteration (group semantics, compatible with `GroupRLDSDataset`) ===
    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        episodes = list(self.episodes)
        if self.train:
            # [Contract] The RLDS path calls `dataset.repeat()` (see vla/datasets/rlds/dataset.py), and
            #   `TrainingStrategy.run_vla_training` explicitly documents that looping over the DataLoader
            #   is "basically infinite" and only terminates on `max_steps`. We MUST mirror that: a *finite*
            #   IterableDataset makes the smaller rank shard (rank1: 25850 batches vs rank0: 25870 for WA01)
            #   run dry first, that rank then exits the loop, and the other rank's next collective blocks
            #   forever => NCCL watchdog kills the whole run. That is exactly the two WA01 crashes that hung
            #   at tqdm position 25850. So: loop forever, re-shuffling episodes every pass (seeded RNG =>
            #   deterministic). The training loop still ends cleanly at `max_steps`.
            while True:
                rng.shuffle(episodes)  # Shuffle *episode* order only; frame order within an episode is preserved.
                yield from self._iter_pass(episodes, rng)
        else:
            rng.shuffle(episodes)
            yield from self._iter_pass(episodes, rng)

    def _iter_pass(self, episodes: List[Dict[str, Any]], rng: np.random.Generator):
        """Yield every frame (group semantics) for one full pass over `episodes`."""
        for ep in episodes:
            # A corrupt / hang-prone AV1 stream must never take down the whole training run: retry once,
            #   then skip the episode. This is the documented root-cause of a NCCL watchdog kill (a rank
            #   stalls in the main-process video decode and the peer watchdog fires after `NCCL_TIMEOUT`).
            # `run_with_timeout` additionally guarantees a hung decode (no exception, just never returns)
            #   raises `DecodeTimeoutError` after `self.decode_timeout` s, so even a *hang* is skipped.
            try:
                actions = self._read_actions(ep["data_path"])      # (L, A) raw
                frames = run_with_timeout(self._decode_video, self.decode_timeout, ep["video_path"])  # (L, H, W, 3) uint8
            except Exception as e:
                overwatch.warning(f"[LeRobot] load failed ep={ep['episode_index']}: {e}; retrying once")
                try:
                    actions = self._read_actions(ep["data_path"])
                    frames = run_with_timeout(self._decode_video, self.decode_timeout, ep["video_path"])
                except Exception as e2:
                    overwatch.warning(f"[LeRobot] skipping ep={ep['episode_index']} after retry: {e2}")
                    continue
            L = min(actions.shape[0], frames.shape[0], ep["length"])
            if L < self.group_size:
                # Too short to form a single group; skip (safety, should not happen for WA01).
                continue

            acts = self._normalize_actions(actions[:L])        # (L, A) normalized
            frames = frames[:L]
            instruction_bytes = ep["instruction"].encode()
            episode_index = ep["episode_index"]

            num_groups = L // self.group_size
            for g in range(num_groups):
                for k in range(self.group_size):
                    i = g * self.group_size + k

                    # Action window [i : i + action_window_size]; pad the tail with the last action and mask.
                    win = acts[i : i + self.action_window_size]  # (<=16, A)
                    win_mask = np.ones(self.action_window_size, dtype=bool)
                    if win.shape[0] < self.action_window_size:
                        pad = self.action_window_size - win.shape[0]
                        win = np.concatenate([win, np.tile(win[-1:], (pad, 1))], axis=0)
                        win_mask[-pad:] = False

                    raw = dict(
                        observation={
                            "image_primary": frames[i][None],  # (1, H, W, 3)
                            "timestep": np.array([i]),         # frame index within the episode
                        },
                        task={"language_instruction": instruction_bytes},
                        action=win.astype(np.float32),         # (16, A)
                        action_mask=win_mask,                  # (16,)
                        dataset_name=self.dataset_name,
                    )

                    frame = self.batch_transform(raw)
                    frame["episode_ids"] = np.array([episode_index])
                    yield frame

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
