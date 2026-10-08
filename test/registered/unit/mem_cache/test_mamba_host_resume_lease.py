"""CPU checks for Mamba resume pins.

Pins have to follow the session tracker's enabled, closed, and generation
rules. These cases use stub nodes and a stub LRU, and they call the real
pin, close, and fork methods.
"""

import unittest
from collections import defaultdict
from types import SimpleNamespace

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams, MatchResult
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
    MambaComponent,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.session_ref_tracker import (
    UnifiedSessionRefTracker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MAMBA = ComponentType.MAMBA


class _LRU:
    def __init__(self):
        self.nodes = set()

    def in_list(self, node):
        return node in self.nodes

    def remove_node(self, node):
        self.nodes.discard(node)

    def insert_mru(self, node):
        self.nodes.add(node)

    def reset_node_mru(self, node):
        self.nodes.add(node)


class _Bridge:
    def __init__(self, comp):
        self.comp = comp

    def reset_session_state(self):
        return None

    def release_session(self, session_id):
        return self.comp.release_session(session_id)


def _cd(host=True):
    return SimpleNamespace(
        host_value=[1] if host else None,
        value=None,
        host_lock_ref=0,
        lock_ref=0,
        session_ref=0,
        session_ids=None,
    )


class _Node:
    def __init__(self, node_id, parent, host=True):
        self.id = node_id
        self.parent = parent
        self.key = None
        self.children = {}
        self.component_data = {MAMBA: _cd(host)}

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return isinstance(other, _Node) and other.id == self.id


def _node(node_id, parent, host=True):
    return _Node(node_id, parent, host)


class ResumeLeaseTests(unittest.TestCase):
    def setUp(self):
        self.freed = []
        self.comp = MambaComponent.__new__(MambaComponent)
        self.comp.component_type = MAMBA
        self.comp.mamba_checkpoint_grid = 1
        self.comp._resume_leases = {}
        self.comp._resume_pins = {}
        self.comp._pending_resume_backup = {}
        self.comp._session_leaves = defaultdict(set)
        pool = SimpleNamespace(size=4, available_size=lambda: 3)
        self.comp.cache = SimpleNamespace(
            session_refs=None,
            _free_values=lambda *_args, **_kwargs: None,
            host_pool_group=SimpleNamespace(get_pool=lambda _name: pool),
        )
        self.root = _Node(0, None, host=False)
        self.nodes = {0: self.root}
        lru = _LRU()
        self.comp.tree_core = SimpleNamespace(
            root_node=self.root,
            enable_session_radix_cache=True,
            host_lru_lists={MAMBA: lru},
            lru_lists={MAMBA: _LRU()},
            evictable_host_leaves=set(),
            _update_evictable_leaf_sets=lambda _node: None,
            _evict_component_and_detach_lru=self._evict,
            _cascade_evict=lambda *_args, **_kwargs: None,
            node_by_id=lambda node_id: self.nodes[node_id],
        )
        self.bridge = _Bridge(self.comp)
        self.tracker = UnifiedSessionRefTracker(
            components=(self.bridge,),
            tree_core=self.comp.tree_core,
            enable_session_radix_cache=True,
        )
        self.comp.cache.session_refs = self.tracker

    def _evict(self, node, _comp, **_kwargs):
        node.component_data[MAMBA].host_value = None
        self.freed.append(node.id)

    def _add(self, node):
        self.nodes[node.id] = node
        return node

    def _req(self, session_id, generation, streaming=False):
        session = SimpleNamespace(streaming=streaming, session_id=session_id)
        return SimpleNamespace(
            session_id=session_id,
            session=session,
            session_generation=generation,
        )

    def _insert(self, node):
        return SimpleNamespace(last_device_node=node.id)

    def _open(self, session_id="s"):
        return self.tracker.open_radix_session(session_id)

    def test_disabled_session_cache_does_not_pin(self):
        self.tracker.enable_session_radix_cache = False
        self.comp.tree_core.enable_session_radix_cache = False
        node = self._add(_node(1, self.root))
        generation = 1
        self.tracker._session_generations["s"] = generation
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(self.comp._resume_pins, {})
        self.assertEqual(self.tracker.release_radix_session("s"), 0)

    def test_close_drops_the_pin_and_a_late_finish_does_not_restore_it(self):
        generation = self._open()
        node = self._add(_node(1, self.root))
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)
        self.tracker.release_radix_session("s")
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn("s", self.comp._resume_pins)

        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.tracker.release_radix_session("s")
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)

    def test_reopened_session_id_ignores_the_previous_generation(self):
        first = self._open()
        node = self._add(_node(1, self.root))
        old = self._req("s", first)
        self.comp._note_inserted_resume(old, self._insert(node))
        self.tracker.release_radix_session("s")
        second = self._open()
        self.assertNotEqual(first, second)

        self.comp._note_inserted_resume(old, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)

        fresh = self._req("s", second)
        self.comp._note_inserted_resume(fresh, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)
        self.comp._note_inserted_resume(old, self._insert(node))
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 1)

    def test_close_clears_match_and_commit_pins(self):
        generation = self._open()
        req = self._req("s", generation)
        match = self._add(_node(1, self.root))
        commit = self._add(_node(2, self.root))
        params = MatchPrefixParams(key=RadixKey(token_ids=[]), req=req)
        result = MatchResult(
            device_indices=[],
            last_device_node=match,
            last_host_node=match,
            best_match_node=match,
        )
        self.comp.finalize_match_result_in_tree_core(result, params, [], 0)
        self.comp._note_inserted_resume(req, self._insert(commit))
        self.assertEqual(match.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(commit.component_data[MAMBA].host_lock_ref, 1)
        self.assertEqual(set(self.comp._resume_pins["s"]), {"match", "commit"})

        self.tracker.release_radix_session("s")
        self.assertEqual(match.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(commit.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn("s", self.comp._resume_pins)

    def test_fork_drops_the_abandoned_tail_and_keeps_the_shared_node(self):
        generation = self._open()
        fork = self._add(_node(1, self.root))
        old = self._add(_node(2, fork))
        new = self._add(_node(3, fork))
        fork.children = {2: old, 3: new}
        old.component_data[MAMBA].session_ids = {"s"}
        self.comp._session_leaves["s"].add(old)

        req = self._req("s", generation)
        self.assertEqual(self.comp._pin_session_id(req), "s")
        self.comp.register_session_leaf("s", new)

        self.assertIsNone(old.component_data[MAMBA].host_value)
        self.assertEqual(fork.component_data[MAMBA].host_value, [1])
        self.assertEqual(self.freed, [old.id])
        self.assertIn(new, self.comp._session_leaves["s"])
        self.assertNotIn(old, self.comp._session_leaves["s"])

    def test_late_host_backup_does_not_pin_a_closed_session(self):
        generation = self._open()
        node = self._add(_node(1, self.root, host=False))
        node.component_data[MAMBA].value = [1]
        req = self._req("s", generation)
        self.comp._note_inserted_resume(req, self._insert(node))
        self.assertEqual(
            self.comp._pending_resume_backup[node.id], ("s", generation)
        )
        self.tracker.release_radix_session("s")
        self.assertNotIn(node.id, self.comp._pending_resume_backup)

        # A backup that still holds the old session id must not pin after close.
        self.comp._pending_resume_backup[node.id] = ("s", generation)
        node.component_data[MAMBA].host_value = [1]
        self.comp.commit_hicache_transfer(
            node,
            CacheTransferPhase.BACKUP_HOST,
            transfers=[
                SimpleNamespace(host_indices=SimpleNamespace(clone=lambda: [1]))
            ],
            cache_actions=[],
        )
        self.assertEqual(node.component_data[MAMBA].host_lock_ref, 0)
        self.assertNotIn(node.id, self.comp._pending_resume_backup)


if __name__ == "__main__":
    unittest.main()
