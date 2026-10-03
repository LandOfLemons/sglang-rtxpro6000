import fcntl
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

from sglang.srt.utils.cpp_extension_loader import load_extension_with_recovery
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_stale_torch_lock_is_removed_before_loading(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    torch_lock_path = build_directory / "lock"
    torch_lock_path.touch()

    expected = object()
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", return_value=expected) as load,
    ):
        result = load_extension_with_recovery("test_extension", ["source.cpp"])

    assert result is expected
    assert not torch_lock_path.exists()
    load.assert_called_once_with(
        name="test_extension",
        sources=["source.cpp"],
        extra_cflags=None,
        extra_cuda_cflags=None,
        extra_ldflags=None,
        build_directory=str(build_directory),
        with_cuda=None,
        verbose=False,
    )


def test_link_flags_and_cuda_toggle_reach_torch(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", return_value=object()) as load,
    ):
        load_extension_with_recovery(
            "test_extension",
            ["source.cpp"],
            extra_ldflags=["-lcrypto"],
            with_cuda=False,
        )

    kwargs = load.call_args.kwargs
    assert kwargs["extra_ldflags"] == ["-lcrypto"]
    assert kwargs["with_cuda"] is False


def test_broken_extension_is_rebuilt_under_the_same_lock(tmp_path: Path):
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    expected = object()
    load_error = OSError(f"{build_directory}/test_extension.so: file too short")

    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch(
            "torch.utils.cpp_extension.load",
            side_effect=[load_error, expected],
        ) as load,
    ):
        result = load_extension_with_recovery("test_extension", ["source.cpp"])

    assert result is expected
    assert build_directory.is_dir()
    assert load.call_count == 2


def test_live_build_holding_the_flock_keeps_its_own_torch_lock(tmp_path: Path):
    """A build in flight owns the flock, so its PyTorch lock file must survive."""
    build_directory = tmp_path / "test_extension"
    build_directory.mkdir()
    torch_lock_path = build_directory / "lock"
    torch_lock_path.touch()

    # Stand-in for the compiler process that created torch's lock file: another
    # open file description holding the advisory build lock.
    holder = (build_directory.parent / ".test_extension.sglang.lock").open("a+")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)

    results = []
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", return_value=object()) as load,
    ):
        worker = threading.Thread(
            target=lambda: results.append(
                load_extension_with_recovery("test_extension", ["source.cpp"])
            )
        )
        worker.start()
        worker.join(1.0)

        assert worker.is_alive(), "a waiter must block behind the live build"
        assert load.call_count == 0
        assert torch_lock_path.exists(), "the live build's lock was stolen"

        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()
        worker.join(60)

    assert not worker.is_alive()
    assert len(results) == 1
    assert load.call_count == 1
    assert not torch_lock_path.exists()


def test_native_hash_loads_under_the_recovering_lock(tmp_path: Path):
    """The HiCache hash extension must reach torch through the recovering loader."""
    from sglang.srt.mem_cache.cpp_utils import native_hash

    build_directory = tmp_path / "hicache_hash_cpp"
    build_directory.mkdir(parents=True)
    stale_lock = build_directory / "lock"
    stale_lock.touch()  # a dead process left it behind mid-build
    lock_seen_by_torch = []

    def _fake_build(*args, **kwargs):
        lock_seen_by_torch.append(stale_lock.exists())
        return object()

    native_hash._load_native_hash_module.cache_clear()
    with (
        patch(
            "sglang.srt.utils.cpp_extension_loader._get_build_directory",
            return_value=build_directory,
        ),
        patch("torch.utils.cpp_extension.load", side_effect=_fake_build) as load,
    ):
        try:
            assert native_hash._load_native_hash_module() is not None
            # torch must never be handed a build directory that still has a lock.
            assert lock_seen_by_torch == [False]
            kwargs = load.call_args.kwargs
            assert kwargs["name"] == "hicache_hash_cpp"
            assert kwargs["with_cuda"] is False
            assert kwargs["extra_ldflags"] == ["-lcrypto"]
            assert kwargs["build_directory"] == str(build_directory)
            assert set(kwargs["extra_cflags"]) >= {"-O3", "-std=c++17", "-DNDEBUG"}
            assert not stale_lock.exists()
        finally:
            native_hash._load_native_hash_module.cache_clear()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
