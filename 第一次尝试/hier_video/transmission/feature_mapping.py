"""Deterministic group-to-region mapping for public and individual features."""

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


Shape5 = Tuple[int, int, int, int, int]


@dataclass(frozen=True)
class TensorRegion:
    """Half-open BTCHW bounds; ``[start,end)`` follows Python slicing."""

    b: Tuple[int, int]
    t: Tuple[int, int]
    c: Tuple[int, int]
    h: Tuple[int, int]
    w: Tuple[int, int]

    @property
    def shape(self) -> Shape5:
        return tuple(end - start for start, end in (self.b, self.t, self.c, self.h, self.w))  # type: ignore[return-value]

    def as_dict(self) -> dict:
        return {
            "b": list(self.b),
            "t": list(self.t),
            "c": list(self.c),
            "h": list(self.h),
            "w": list(self.w),
            "shape": list(self.shape),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TensorRegion":
        return cls(
            b=tuple(value["b"]),
            t=tuple(value["t"]),
            c=tuple(value["c"]),
            h=tuple(value["h"]),
            w=tuple(value["w"]),
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class FeatureGroup:
    group_id: int
    branch: str
    tensor_region: TensorRegion
    priority: int
    required: bool

    def as_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "branch": self.branch,
            "layout": "BTCHW",
            "tensor_region": self.tensor_region.as_dict(),
            "priority": self.priority,
            "required": self.required,
        }


@dataclass(frozen=True)
class FeatureGroupManifest:
    interface_version: str
    layout: str
    public_shape: Shape5
    individual_shape: Shape5
    tile_hw: Tuple[int, int]
    groups: Tuple[FeatureGroup, ...]

    def as_dict(self) -> dict:
        return {
            "interface_version": self.interface_version,
            "layout": self.layout,
            "public_shape": list(self.public_shape),
            "individual_shape": list(self.individual_shape),
            "tile_hw": list(self.tile_hw),
            "groups": [group.as_dict() for group in self.groups],
        }

    def validate(self) -> None:
        if self.layout != "BTCHW":
            raise ValueError(f"Expected BTCHW layout, got {self.layout!r}")
        if [g.group_id for g in self.groups] != list(range(len(self.groups))):
            raise ValueError("group_id values must be contiguous and start at zero")
        for branch, shape in (("public", self.public_shape), ("individual", self.individual_shape)):
            branch_groups = [g for g in self.groups if g.branch == branch]
            if not branch_groups:
                raise ValueError(f"No groups found for branch {branch!r}")
            for group in branch_groups:
                region = group.tensor_region
                for bounds, limit in zip((region.b, region.t, region.c, region.h, region.w), shape):
                    start, end = bounds
                    if not (0 <= start < end <= limit):
                        raise ValueError(f"Group {group.group_id} is outside {branch} shape {shape}")
            for index, left in enumerate(branch_groups):
                for right in branch_groups[index + 1 :]:
                    if _regions_overlap(left.tensor_region, right.tensor_region):
                        raise ValueError(
                            f"Overlapping group regions: {left.group_id} and {right.group_id}"
                        )
            covered = sum(_region_volume(group.tensor_region) for group in branch_groups)
            expected = _shape_volume(shape)
            if covered != expected:
                raise ValueError(f"Groups do not fully cover {branch}: {covered} != {expected}")


def _shape_volume(shape: Sequence[int]) -> int:
    volume = 1
    for value in shape:
        volume *= int(value)
    return volume


def _region_volume(region: TensorRegion) -> int:
    return _shape_volume(region.shape)


def _regions_overlap(left: TensorRegion, right: TensorRegion) -> bool:
    for left_bounds, right_bounds in zip(
        (left.b, left.t, left.c, left.h, left.w),
        (right.b, right.t, right.c, right.h, right.w),
    ):
        if max(left_bounds[0], right_bounds[0]) >= min(left_bounds[1], right_bounds[1]):
            return False
    return True


def _starts(size: int, tile: int) -> Iterable[tuple[int, int]]:
    for start in range(0, size, tile):
        yield start, min(start + tile, size)


def _groups_for_branch(
    branch: str,
    shape: Shape5,
    tile_hw: tuple[int, int],
    first_group_id: int,
    priority: int,
    required: bool,
) -> list[FeatureGroup]:
    batch, frames, channels, height, width = shape
    tile_h, tile_w = tile_hw
    groups: list[FeatureGroup] = []
    next_id = first_group_id
    for b in range(batch):
        for t in range(frames):
            for h_bounds in _starts(height, tile_h):
                for w_bounds in _starts(width, tile_w):
                    groups.append(
                        FeatureGroup(
                            group_id=next_id,
                            branch=branch,
                            tensor_region=TensorRegion(
                                b=(b, b + 1),
                                t=(t, t + 1),
                                c=(0, channels),
                                h=h_bounds,
                                w=w_bounds,
                            ),
                            priority=priority,
                            required=required,
                        )
                    )
                    next_id += 1
    return groups


def build_group_manifest(
    public_shape: Sequence[int],
    individual_shape: Sequence[int],
    *,
    tile_hw: tuple[int, int] = (16, 16),
    interface_version: str = "v1",
) -> FeatureGroupManifest:
    public = tuple(int(value) for value in public_shape)
    individual = tuple(int(value) for value in individual_shape)
    if len(public) != 5 or len(individual) != 5:
        raise ValueError("Both shapes must be rank-5 BTCHW")
    if public[0] != individual[0] or public[2] != individual[2]:
        raise ValueError("Public and individual B/C dimensions must match")
    if any(value < 1 for value in public + individual):
        raise ValueError("All shape values must be positive")
    if len(tile_hw) != 2 or any(int(value) < 1 for value in tile_hw):
        raise ValueError("tile_hw must contain two positive integers")
    tile_hw = (int(tile_hw[0]), int(tile_hw[1]))
    public_groups = _groups_for_branch("public", public, tile_hw, 0, 0, True)
    individual_groups = _groups_for_branch(
        "individual", individual, tile_hw, len(public_groups), 1, False
    )
    manifest = FeatureGroupManifest(
        interface_version=interface_version,
        layout="BTCHW",
        public_shape=public,  # type: ignore[arg-type]
        individual_shape=individual,  # type: ignore[arg-type]
        tile_hw=tile_hw,
        groups=tuple(public_groups + individual_groups),
    )
    manifest.validate()
    return manifest
