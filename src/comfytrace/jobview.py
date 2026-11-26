"""Group report evidence by recovered prompt ID and verified output position."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Output:
    handoffs: set[str] = field(default_factory=set)
    images: set[str] = field(default_factory=set)
    physical: list[tuple[int, int]] = field(default_factory=list)
    pixels: dict[tuple[str, int], list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    frame_sizes: dict[tuple[str, int], int] = field(default_factory=dict)


@dataclass
class Job:
    workflows: set[str] = field(default_factory=set)
    findings: set[str] = field(default_factory=set)
    images: set[str] = field(default_factory=set)
    outputs: dict[tuple[str, int], Output] = field(default_factory=lambda: defaultdict(Output))


def interval(offset: object, length: object) -> tuple[int, int] | None:
    if type(offset) is int and type(length) is int and offset >= 0 and length > 0:
        return offset, offset + length
    return None


def union_size(spans: list[tuple[int, int]]) -> int:
    total = end = 0
    for start, stop in sorted(spans):
        if stop > end:
            total += stop - max(start, end)
            end = stop
    return total
