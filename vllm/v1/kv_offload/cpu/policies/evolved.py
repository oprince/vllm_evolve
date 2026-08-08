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

    S3-FIFO / SIEVE hybrid with capped visited counters and decaying second
    chances, plus a bounded ghost list for warm re-admission of recurring
    multi-turn prefixes. One-shot decode blocks (freq 0) are evicted quickly;
    recurring prefixes accumulate protection and, if evicted then re-accessed,
    re-enter with a small warm credit of 1.
    """

    # EVOLVE-BLOCK-START

    @override
    def __init__(self, cache_capacity: int) -> None:
        """Init FIFO ring, visited-frequency counters, and a small ghost list."""
        self._capacity = cache_capacity
        self.evictable_blocks: OrderedDict[OffloadKey, None] = OrderedDict()
        self.blocks: dict[OffloadKey, BlockStatus] = {}
        self._freq: dict[OffloadKey, int] = {}
        self._MAX_FREQ = 3
        # Ghost list: recently-evicted keys, LRU-ordered, bounded overhead.
        self._ghost: OrderedDict[OffloadKey, None] = OrderedDict()
        self._ghost_cap = max(cache_capacity, 1)

    @override
    def get(self, key: OffloadKey) -> BlockStatus | None:
        return self.blocks.get(key)

    @override
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Track block; if evictable, enter FIFO ring. If key was recently
        evicted (in ghost list), it is a proven re-access of a recurring
        prefix, so warm-start its frequency to 1 instead of 0. A modest
        credit (1) protects recurring prefixes without over-holding them,
        which empirically maximizes hit rate and minimizes TTFT."""
        self.blocks[key] = block
        if block.ref_cnt == 0:
            self.evictable_blocks[key] = None
            if key not in self._freq:
                if self._ghost.pop(key, None) is not None:
                    self._freq[key] = 1  # warm re-admission
                else:
                    self._freq[key] = 0

    @override
    def remove(self, key: OffloadKey) -> None:
        del self.blocks[key]
        self.evictable_blocks.pop(key, None)
        self._freq.pop(key, None)

    @override
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """Bump visited counter only (SIEVE-style, no list reordering)."""
        freq = self._freq
        cap = self._MAX_FREQ
        for key in keys:
            f = freq.get(key)
            if f is not None and f < cap:
                freq[key] = f + 1

    @override
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """S3-FIFO/SIEVE eviction: visited blocks get a decaying second chance;
        truly evicted keys are recorded in the bounded ghost list so a future
        re-access can be recognized as a recurring prefix."""
        if n == 0:
            return []

        candidates: list[tuple[OffloadKey, BlockStatus]] = []
        freq = self._freq
        evictable = self.evictable_blocks
        scan_limit = len(evictable) + n + 1
        scanned = 0
        while len(candidates) < n and evictable and scanned < scan_limit:
            scanned += 1
            key = next(iter(evictable))
            if key in protected:
                evictable.move_to_end(key)
                continue
            f = freq.get(key, 0)
            if f > 0:
                freq[key] = f - 1
                evictable.move_to_end(key)
                continue
            block = self.blocks[key]
            assert block.ref_cnt == 0
            candidates.append((key, block))
            del evictable[key]

        if len(candidates) < n:
            for key, block in candidates:
                evictable[key] = None
            return None

        ghost = self._ghost
        ghost_cap = self._ghost_cap
        for key, _ in candidates:
            del self.blocks[key]
            freq.pop(key, None)
            ghost[key] = None
            if len(ghost) > ghost_cap:
                ghost.popitem(last=False)
        return candidates

    @override
    def clear(self) -> None:
        self.evictable_blocks.clear()
        self.blocks.clear()
        self._freq.clear()
        self._ghost.clear()

    @override
    def mark_evictable(self, key: OffloadKey) -> None:
        self.evictable_blocks[key] = None
        self._freq.setdefault(key, 0)

    @override
    def mark_non_evictable(self, key: OffloadKey) -> None:
        del self.evictable_blocks[key]

    # EVOLVE-BLOCK-END
