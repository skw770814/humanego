from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from ego_relation.config import ProjectConfig
from ego_relation.contracts.se3 import compose, invert
from ego_relation.s2_object_relations.encoding import build_relation_tokens, relation_text

PACKET_SCHEMA = "ego_relation_observation_v1"


class OnlineRelationFormatter:
    """Stateful visual/FK fusion with the same token contract as offline data.

    The detector/tracker supplies object poses in the robot camera frame. Robot
    FK supplies both hand poses in that same frame. This class owns grasp latch,
    finite relation text, token ordering and atomic packet publication.
    """

    def __init__(
        self,
        cfg: ProjectConfig,
        categories: list[str],
        is_anchor: np.ndarray,
        initial_object_poses: np.ndarray,
    ) -> None:
        self.cfg = cfg
        self.categories = np.asarray(categories)
        self.is_anchor = np.asarray(is_anchor, dtype=bool)
        self.objects = np.asarray(initial_object_poses, dtype=np.float64).copy()
        if self.objects.shape != (len(self.categories), 4, 4):
            raise ValueError(f"Expected object poses {(len(self.categories), 4, 4)}, got {self.objects.shape}")
        if self.is_anchor.shape != (len(self.categories),):
            raise ValueError("is_anchor shape does not match categories")
        self._owner = [-1, -1]
        self._offset: list[np.ndarray | None] = [None, None]
        self._previous_grasp = [False, False]
        self._previous_distances: np.ndarray | None = None
        self._previous_timestamp_ns: int | None = None
        self._sequence = 0

    def update(
        self,
        *,
        timestamp_unix_ns: int,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        left_grasp: bool,
        right_grasp: bool,
        visual_object_poses: np.ndarray,
        visual_valid: np.ndarray,
    ) -> dict:
        timestamp_unix_ns = int(timestamp_unix_ns)
        if self._previous_timestamp_ns is not None and timestamp_unix_ns <= self._previous_timestamp_ns:
            raise ValueError("Online relation timestamp must be strictly increasing")
        hands = [np.asarray(left_hand, dtype=np.float64), np.asarray(right_hand, dtype=np.float64)]
        grasps = [bool(left_grasp), bool(right_grasp)]
        visual = np.asarray(visual_object_poses, dtype=np.float64)
        visual_valid = np.asarray(visual_valid, dtype=bool)
        if visual.shape != self.objects.shape or visual_valid.shape != (len(self.objects),):
            raise ValueError("Online visual object pose shape mismatch")
        if not all(hand.shape == (4, 4) for hand in hands):
            raise ValueError("Online hand poses must both be 4x4")

        claimed = {index for index in self._owner if index >= 0}
        for index in range(len(self.objects)):
            if index not in claimed and visual_valid[index]:
                self.objects[index] = visual[index]

        dynamic = np.zeros(len(self.objects), dtype=bool)
        for hand_index, (hand, grasp) in enumerate(zip(hands, grasps, strict=True)):
            if grasp and not self._previous_grasp[hand_index]:
                distances = np.linalg.norm(self.objects[:, :3, 3] - hand[:3, 3], axis=1)
                candidates = [int(index) for index in np.argsort(distances) if int(index) not in claimed]
                if candidates and distances[candidates[0]] <= self.cfg.perception.latch_distance_m:
                    owner = candidates[0]
                    self._owner[hand_index] = owner
                    self._offset[hand_index] = compose(invert(hand), self.objects[owner])
                    claimed.add(owner)
            elif not grasp and self._previous_grasp[hand_index]:
                owner = self._owner[hand_index]
                if owner >= 0:
                    claimed.discard(owner)
                self._owner[hand_index] = -1
                self._offset[hand_index] = None
            self._previous_grasp[hand_index] = grasp
            owner = self._owner[hand_index]
            if grasp and owner >= 0:
                assert self._offset[hand_index] is not None
                self.objects[owner] = compose(hand, self._offset[hand_index])
                dynamic[owner] = True

        tokens, mask = build_relation_tokens(
            hands[0],
            hands[1],
            self.objects,
            self.is_anchor,
            grasps[0],
            grasps[1],
            dynamic,
            max_entities=self.cfg.perception.max_entities,
            translation_scale_m=self.cfg.relations.translation_scale_m,
        )
        dt = (
            1.0 / self.cfg.realtime.tracker_hz
            if self._previous_timestamp_ns is None
            else (timestamp_unix_ns - self._previous_timestamp_ns) / 1e9
        )
        event_text, self._previous_distances = relation_text(
            self.categories,
            self.objects,
            hands[0],
            hands[1],
            self._previous_distances,
            dt,
            self.cfg,
        )
        packet = {
            "schema": PACKET_SCHEMA,
            "sequence": self._sequence,
            "timestamp_unix_ns": timestamp_unix_ns,
            "relation_schema": "humanego_dual_hand_ict_v1",
            "relation_tokens": tokens.tolist(),
            "relation_mask": mask.tolist(),
            "event_text": event_text,
            "object_dynamic": dynamic.tolist(),
            "object_owner": self._owner.copy(),
        }
        self._sequence += 1
        self._previous_timestamp_ns = timestamp_unix_ns
        return packet

    @staticmethod
    def publish_atomic(path: str | Path, packet: dict) -> None:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp")
        temporary.write_text(json.dumps(packet, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, destination)
