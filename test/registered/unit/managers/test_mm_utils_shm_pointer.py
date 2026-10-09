"""Unit tests for ``ShmPointerMMData`` when its segment disappears in transit.

Multimodal features travel tokenizer -> scheduler as POSIX shared-memory
pointers. If the segment is unlinked before a receiver attaches (another
consumer materialized it first, a leaked-segment sweep, or a lost race between
TP ranks), the receiver used to raise FileNotFoundError from inside
``recv_pyobj`` and take the whole scheduler process down with it
(``sgl_shm_mm_<pid>_<rand>`` crashes under image-heavy agentic load).

The pointer must now survive unpickling and report the loss on
``materialize()`` so the request receiver can reject just that request.
No server / GPU / weight loading involved.
"""

import pickle
import unittest

import msgspec
from multiprocessing import shared_memory

import torch

from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.mm_utils import (
    ShmPointerMMData,
    discard_shm_features,
    has_shm_features,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _segment_exists(name: str) -> bool:
    try:
        shm = shared_memory.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    shm.close()
    return True


class TestShmPointerMMData(CustomTestCase):
    def test_roundtrip_materializes_and_unlinks(self):
        src = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        ptr = ShmPointerMMData(src, precomputed_hash=42)
        self.assertTrue(_segment_exists(ptr.shm_name))

        received = pickle.loads(pickle.dumps(ptr))
        self.assertIsNone(received._materialization_error)
        out = received.materialize()
        torch.testing.assert_close(out, src)
        self.assertEqual(received.precomputed_hash, 42)
        self.assertFalse(_segment_exists(ptr.shm_name))
        self.assertIsNone(received._shm_handle)

    def test_missing_segment_does_not_raise_on_unpickle(self):
        src = torch.ones(16, dtype=torch.float16)
        ptr = ShmPointerMMData(src)
        payload = pickle.dumps(ptr)
        # Simulate the lost race: the segment is gone before this receiver attaches.
        shared_memory.SharedMemory(name=ptr.shm_name).unlink()

        received = pickle.loads(payload)  # must not raise
        self.assertIsNotNone(received._materialization_error)
        self.assertIn("FileNotFoundError", received._materialization_error)
        self.assertIsNone(received.tensor)

        with self.assertRaises(RuntimeError) as ctx:
            received.materialize()
        self.assertIn(ptr.shm_name, str(ctx.exception))
        # cleanup is idempotent on a dead segment
        received.close_and_unlink()

    def test_second_receiver_after_materialize_is_rejected_not_fatal(self):
        src = torch.zeros(8, dtype=torch.float32)
        ptr = ShmPointerMMData(src)
        payload = pickle.dumps(ptr)
        first = pickle.loads(payload)
        first.materialize()  # unlinks
        second = pickle.loads(payload)  # the double-consumption case
        self.assertIsNotNone(second._materialization_error)
        with self.assertRaises(RuntimeError):
            second.materialize()

    def test_discard_shm_features_releases_segments(self):
        feats = [ShmPointerMMData(torch.zeros(4)), ShmPointerMMData(torch.zeros(4))]
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 2), (2, 4)],
            feature=feats,
        )
        req = _tokenized_req(mm_inputs=_FakeMMInputs([item]))
        self.assertTrue(has_shm_features([req]))
        names = [f.shm_name for f in feats]
        discard_shm_features(req)
        for name in names:
            self.assertFalse(_segment_exists(name))


def _tokenized_req(**overrides) -> TokenizedGenerateReqInput:
    """Build a TokenizedGenerateReqInput with every required field defaulted to None."""
    kwargs = {
        f.name: None for f in msgspec.structs.fields(TokenizedGenerateReqInput) if f.required
    }
    kwargs.update(rid="r1", input_text="", input_ids=[1, 2, 3])
    kwargs.update(overrides)
    return TokenizedGenerateReqInput(**kwargs)


class _FakeMMInputs:
    """Minimal stand-in exposing ``mm_items`` like MultimodalProcessorOutput."""

    def __init__(self, mm_items):
        self.mm_items = mm_items


if __name__ == "__main__":
    unittest.main()
