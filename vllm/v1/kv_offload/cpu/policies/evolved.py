# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Iterable

from typing_extensions import override

from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy


class EVCachePolicy(CachePolicy):
    """
    Evolved cache policy for KV offloading.

    Starting point: LRU eviction with an evictable list for O(1) eviction.
    """

    # EVOLVE-BLOCK-START

    @override
    def __init__(self, cache_capacity: int) -> None:
        self._capacity = cache_capacity
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        for key in reversed(list(keys)):
            if key in self.evictable_blocks:
                self.evictable_blocks.move_to_end(key)

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        for key, _ in self.evictable_blocks.items():
            if key in protected:
                continue
            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))
            if len(candidates) == n:
                break

        if len(candidates) < n:
            return None
        for key, _ in candidates:
            del self.evictable_blocks[key]
            del self.blocks[key]
        return candidates

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks[key] = None

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        del self.evictable_blocks[key]

    # EVOLVE-BLOCK-END