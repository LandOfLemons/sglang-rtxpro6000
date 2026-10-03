"""CPU-only checks for the prebuilt-image launch examples in
docker/pennyroyal/launch: the host run.sh builds one plain docker command from
the operator's own files, and the image entrypoint hands a mounted startup
script to the image's own SGLang with the qualified flags, helpers, namespace
and pinned assets intact.

No GPU, no container runtime, no registry and no model: docker is a capturing
stub and the server binary is a capturing script that records its argv.
"""

import json
import os
import subprocess
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
LAUNCH = REPO / "docker" / "pennyroyal" / "launch"
ENTRYPOINT = REPO / "docker" / "pennyroyal" / "entrypoint.sh"
CONFIGS = LAUNCH / "config"
RECIPES = REPO / "configs" / "pennyroyal"
# Never inherited from the developer's shell: these are exactly what each test
# decides, so a stray export cannot change the expected launch.
CONTROLLED = (
    "PENNYROYAL_IMAGE",
    "PENNYROYAL_STARTUP",
    "PENNYROYAL_NIXL_CONFIG",
    "PENNYROYAL_PORT",
    "PENNYROYAL_USER",
    "HOST_MODELS_ROOT",
    "HOST_CACHE_BASE",
    "HOST_NIXL_STORAGE_BASE",
    "NVIDIA_GPU",
    "TARGET_MODEL",
    "DRAFT_MODEL",
    "CACHE_BASE",
    "NIXL_STORAGE_BASE",
    "NIXL_CONFIG",
    "PENNY_HICACHE_SIZE_GB",
    "PENNY_PLE_BACKEND",
    "TP_SIZE",
    "MAX_RUNNING_REQUESTS",
    "MAX_MAMBA_CACHE_SIZE",
    "MAX_TOTAL_TOKENS",
    "SGLANG_MM_PREPROCESS_DEVICE",
    "SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR",
)


def clean_env(**updates) -> dict[str, str]:
    env = {name: value for name, value in os.environ.items() if name not in CONTROLLED}
    return {**env, **updates}


def argv_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def values_after(args: list[str], flag: str) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args) if value == flag]


def qualified_defaults() -> dict[str, str]:
    # The recipes read these optional knobs from the environment; the mounted
    # scripts state them as settings. Pin the environment to what the recipes
    # would default to, so the two paths can be compared.
    return {
        "PENNY_HICACHE_SIZE_GB": "",
        "PENNY_PLE_BACKEND": "",
        "TP_SIZE": "",
        "MAX_RUNNING_REQUESTS": "",
        "MAX_MAMBA_CACHE_SIZE": "",
        "MAX_TOTAL_TOKENS": "",
        "SGLANG_MM_PREPROCESS_DEVICE": "",
    }


# --------------------------------------------------------------------- run.sh


def fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    """A docker on PATH that records its argv instead of talking to a daemon."""
    capture = tmp_path / f"docker-argv-{os.urandom(4).hex()}"
    directory = tmp_path / "bin"
    directory.mkdir(exist_ok=True)
    stub = directory / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "$DOCKER_CAPTURE"\n'
        'for argument in "$@"; do printf "%s\\n" "$argument" >> "$DOCKER_CAPTURE"; done\n'
    )
    stub.chmod(0o755)
    return directory, capture


def host_dirs(tmp_path: Path):
    roots = {}
    for name in ("models", "cache", "nixl"):
        directory = tmp_path / "host share" / name
        directory.mkdir(parents=True, exist_ok=True)
        roots[name] = directory
    return roots["models"], roots["cache"], roots["nixl"]


def run_launcher(tmp_path: Path, *options: str):
    directory, capture = fake_docker(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCH / "run.sh"), *options],
        env=clean_env(
            PATH=f"{directory}:{os.environ['PATH']}",
            DOCKER_CAPTURE=str(capture),
            HOME=str(tmp_path / "home"),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    argv = capture.read_text().splitlines() if capture.exists() else None
    return result, argv


def test_launcher_mounts_directories_and_runs_the_mounted_script(tmp_path):
    startup = tmp_path / "my scripts" / "start up.sh"
    startup.parent.mkdir()
    startup.write_text("#!/usr/bin/env bash\nexit 0\n")
    nixl_config = tmp_path / "other share" / "nixl conf.toml"
    nixl_config.parent.mkdir()
    nixl_config.write_text("use_direct_io = true\n")
    models, cache, nixl = host_dirs(tmp_path)

    result, argv = run_launcher(
        tmp_path,
        "--startup",
        str(startup),
        "--nixl-config",
        str(nixl_config),
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
        "--image",
        "pennyroyal:test",
        "--port",
        "8099",
        "--gpu",
        "0,1",
        "--user",
        "1000:1000",
    )

    assert result.returncode == 0, result.stderr
    assert argv is not None
    assert argv[0] == "run"
    # Directories, never single files: an editor that writes a new file and
    # renames it must still be visible on the next start.
    assert values_after(argv, "--volume") == [
        f"{startup.parent}:/config:ro",
        f"{models}:/models:ro",
        f"{cache}:/cache",
        f"{nixl}:/nixl",
        f"{nixl_config.parent}:/nixl-config:ro",
    ]
    assert argv_after(argv, "--gpus") == "device=0,1"
    assert argv_after(argv, "--publish") == "8099:8001"
    assert argv_after(argv, "--user") == "1000:1000"
    assert argv_after(argv, "--shm-size") == "16g"
    assert values_after(argv, "--ulimit") == ["memlock=-1:-1"]
    assert values_after(argv, "--security-opt") == ["seccomp=unconfined"]
    # The image's existing exec capability: no --entrypoint, no CMD profile.
    assert argv[-3:] == ["exec", "bash", "/config/start up.sh"]
    assert argv[argv.index("pennyroyal:test") + 1] == "exec"
    # A config outside the script's directory arrives as a container path.
    assert values_after(argv, "-e") == ["NIXL_CONFIG=/nixl-config/nixl conf.toml"]


def test_launcher_default_startup_keeps_the_script_own_nixl_config(tmp_path):
    models, cache, nixl = host_dirs(tmp_path)
    result, argv = run_launcher(
        tmp_path,
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
    )
    assert result.returncode == 0, result.stderr
    assert argv[-3:] == ["exec", "bash", "/config/start-flash-next-frspec.sh"]
    assert values_after(argv, "--volume")[0] == f"{CONFIGS}:/config:ro"
    assert values_after(argv, "--volume")[3] == f"{nixl}:/nixl"
    # Nothing was named, so the selected script keeps the config it names.
    assert "-e" not in argv


def nixl_volumes(argv: list[str]) -> list[str]:
    return [item for item in values_after(argv, "--volume") if item.endswith(":/nixl")]


def test_launcher_nixl_off_skips_the_mount_but_not_nvme_io_uring(tmp_path):
    models, cache, _ = host_dirs(tmp_path)
    common = ("--models", str(models), "--cache", str(cache))

    result, argv = run_launcher(tmp_path, "--no-nixl", *common)
    assert result.returncode == 0, result.stderr
    assert nixl_volumes(argv) == []
    assert "--security-opt" not in argv

    result, argv = run_launcher(tmp_path, "--no-nixl", "--nvme-ple", *common)
    assert result.returncode == 0, result.stderr
    # The independent NVMe PLE reader needs io_uring even with NIXL switched off.
    assert values_after(argv, "--security-opt") == ["seccomp=unconfined"]
    assert nixl_volumes(argv) == []


def test_launcher_refuses_missing_or_unreadable_inputs(tmp_path):
    models, cache, nixl = host_dirs(tmp_path)
    common = (
        "--models",
        str(models),
        "--cache",
        str(cache),
        "--nixl-root",
        str(nixl),
    )

    absent = tmp_path / "absent.sh"
    result, argv = run_launcher(tmp_path, "--startup", str(absent), *common)
    assert result.returncode != 0 and argv is None
    assert "Startup script is missing or unreadable" in result.stderr

    # A directory is not a startup script, and no image default fills in.
    result, argv = run_launcher(tmp_path, "--startup", str(models), *common)
    assert result.returncode != 0 and argv is None
    assert "Startup script is missing or unreadable" in result.stderr

    unreadable = tmp_path / "unreadable.sh"
    unreadable.write_text("exit 0\n")
    unreadable.chmod(0o000)
    if not os.access(unreadable, os.R_OK):  # skip when root or ACLs allow the read
        result, argv = run_launcher(tmp_path, "--startup", str(unreadable), *common)
        assert result.returncode != 0 and argv is None
        assert "Startup script is missing or unreadable" in result.stderr
    unreadable.chmod(0o644)

    absent_toml = tmp_path / "absent.toml"
    result, argv = run_launcher(tmp_path, *common, "--nixl-config", str(absent_toml))
    assert result.returncode != 0 and argv is None
    assert "NIXL config is missing or unreadable" in result.stderr

    missing_nixl_root = tmp_path / "no-nixl-here"
    result, argv = run_launcher(
        tmp_path, *common, "--nixl-root", str(missing_nixl_root)
    )
    assert result.returncode != 0 and argv is None
    assert "NIXL root is missing or not writable" in result.stderr

    cache.chmod(0o500)
    try:
        result, argv = run_launcher(tmp_path, *common, "--cache", str(cache))
        assert result.returncode != 0 and argv is None
        assert "Cache root is missing or not writable" in result.stderr
    finally:
        cache.chmod(0o700)

    result, argv = run_launcher(tmp_path, *common, "--bogus")
    assert result.returncode != 0 and argv is None
    assert "Unknown option" in result.stderr


# --------------------------------------------------- image + mounted startup


def make_checkpoint(path: Path, marker: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"marker": marker}))
    (path / "tokenizer.json").write_text(json.dumps({"marker": marker}))
    weights = path / "model.safetensors"
    if not weights.is_file():
        header = json.dumps(
            {"tensor": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
        ).encode()
        weights.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"marker": marker}})
    )
    return path


def target_checkpoint(tmp_path: Path) -> Path:
    return make_checkpoint(tmp_path / "model-share", "target")


def draft_checkpoint(tmp_path: Path) -> Path:
    return make_checkpoint(tmp_path / "model-share" / "draft", "draft")


def image_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A stand-in /opt/pennyroyal: the real helpers reached through symlinks,
    the venv binaries replaced by recorders, one commit for the namespace
    identity. Built once so two launches in a test share the same identity."""
    root = tmp_path / "opt pennyroyal"
    capture = root / "server-argv"
    env_capture = root / "server-env"
    if (root / ".venv" / "bin" / "sglang").exists():
        return root, capture, env_capture
    (root / ".venv" / "bin").mkdir(parents=True)
    # One line answers both launcher probes: the torch version and the visible
    # CUDA device count (TP1 and TP2 both fit, and no GPU is touched).
    fake_python = root / ".venv" / "bin" / "python"
    fake_python.write_text("#!/bin/sh\nprintf 2\n")
    fake_python.chmod(0o755)
    sglang = root / ".venv" / "bin" / "sglang"
    sglang.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ': > "$SGLANG_CAPTURE"\n'
        ': > "$SGLANG_ENV_CAPTURE"\n'
        'for argument in "$@"; do printf "%s\\n" "$argument" >> "$SGLANG_CAPTURE"; done\n'
        'printf "namespace=%s\\n" "${SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
        'printf "cache=%s\\n" "${SGLANG_CACHE_DIR:-unset}" >> "$SGLANG_ENV_CAPTURE"\n'
    )
    sglang.chmod(0o755)
    (root / "configs").symlink_to(REPO / "configs")
    (root / "scripts").symlink_to(REPO / "scripts")
    (root / "VERSION").write_text("stand-in image\n")
    for step in (
        ["git", "init", "-q", str(root)],
        ["git", "-C", str(root), "config", "user.email", "t@example.com"],
        ["git", "-C", str(root), "config", "user.name", "T"],
        ["git", "-C", str(root), "add", "VERSION"],
        ["git", "-C", str(root), "commit", "-qm", "stand-in image"],
    ):
        subprocess.run(step, check=True)
    return root, capture, env_capture


def prepared_config(tmp_path: Path, *names: str, with_nixl_config: bool = True) -> Path:
    """Copy the shipped examples into the operator's own directory and point
    them at the checkpoints that exist here, which is exactly the edit an
    operator makes before starting."""
    directory = tmp_path / "launch config"
    directory.mkdir(exist_ok=True)
    if with_nixl_config:
        for toml in ("nixl-posix.toml", "nixl-posix-frspec.toml"):
            (directory / toml).write_text((CONFIGS / toml).read_text())
    target = str(target_checkpoint(tmp_path))
    draft = str(draft_checkpoint(tmp_path))
    for name in names:
        text = (CONFIGS / name).read_text()
        for old, new in (
            (
                'TARGET_MODEL="/models/RadixArk-Qwen3.8-Flash-Next-NVFP4"',
                f'TARGET_MODEL="{target}"',
            ),
            ('TARGET_MODEL="/models/Qwen3.8-27B-FP8"', f'TARGET_MODEL="{target}"'),
            ('DRAFT_MODEL="/models/Qwen3.8-27B-DFlash2"', f'DRAFT_MODEL="{draft}"'),
        ):
            text = text.replace(old, new)
        assert f'TARGET_MODEL="{target}"' in text, name
        (directory / name).write_text(text)
    return directory


def launch(
    tmp_path: Path,
    startup: Path,
    *,
    through_entrypoint: bool = True,
    keep_nixl_root: bool = False,
    nixl_root: Path | None = None,
    env: dict | None = None,
):
    root, capture, env_capture = image_root(tmp_path)
    capture.unlink(missing_ok=True)
    env_capture.unlink(missing_ok=True)
    cache = tmp_path / "cache root"
    cache.mkdir(exist_ok=True)
    nixl = tmp_path / "nixl root"
    nixl.mkdir(exist_ok=True)
    if nixl_root is not None:
        nixl = nixl_root
    command = (
        ["bash", str(ENTRYPOINT), "exec", "bash", str(startup)]
        if through_entrypoint
        else ["bash", str(startup)]
    )
    result = subprocess.run(
        command,
        env=clean_env(
            REPO_ROOT=str(root),
            CACHE_BASE=str(cache),
            # keep_nixl_root leaves the image's own /nixl default in place.
            NIXL_STORAGE_BASE="/nixl" if keep_nixl_root else str(nixl),
            SGLANG_EXE=str(root / ".venv" / "bin" / "sglang"),
            PYTHON=str(root / ".venv" / "bin" / "python"),
            SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR="",
            HOME=str(tmp_path / "home"),
            SGLANG_CAPTURE=str(capture),
            SGLANG_ENV_CAPTURE=str(env_capture),
            TARGET_MODEL=str(target_checkpoint(tmp_path)),
            DRAFT_MODEL=str(draft_checkpoint(tmp_path)),
            **qualified_defaults(),
            **(env or {}),
        ),
        text=True,
        capture_output=True,
        check=False,
    )
    argv = capture.read_text().splitlines() if capture.exists() else None
    recorded = env_capture.read_text().splitlines() if env_capture.exists() else []
    server_env = dict(line.split("=", 1) for line in recorded)
    return result, argv, server_env, (root, cache, nixl)


def test_examples_are_editable_startup_files_with_container_paths(tmp_path):
    for name in (
        "start-flash-next.sh",
        "start-flash-next-frspec.sh",
        "start-flash-next-no-nixl.sh",
        "start-27b-dflash2.sh",
    ):
        text = (CONFIGS / name).read_text()
        assert 'TARGET_MODEL="/models/' in text, name
        assert 'REPO_ROOT="${REPO_ROOT:-/opt/pennyroyal}"' in text, name
        # Helpers, pinned template and pinned map stay inside the image.
        assert 'source "$IMAGE_CONFIGS/chat-template.sh"' in text, name
        assert 'TARGET_OVERRIDES=\'' in text, name
    assert (LAUNCH / "run.sh").is_file()
    assert (CONFIGS / "nixl-posix.toml").is_file()
    assert (CONFIGS / "nixl-posix-frspec.toml").is_file()


def test_mounted_startup_reaches_the_server_unchanged(tmp_path):
    config = prepared_config(tmp_path, "start-flash-next.sh")
    result, argv, server_env, (root, cache, nixl) = launch(
        tmp_path, config / "start-flash-next.sh"
    )
    assert result.returncode == 0, result.stderr
    assert argv is not None and argv[0] == "serve"
    # Qualified launch flags, exactly as the in-image recipe passes them.
    for flag, value in (
        ("--quantization", "modelopt_fp4"),
        ("--kv-cache-dtype", "fp8_e4m3"),
        ("--mem-fraction-static", "0.981"),
        ("--context-length", "524288"),
        ("--page-size", "64"),
        ("--mamba-track-interval", "64"),
        ("--max-running-requests", "4"),
        ("--max-mamba-cache-size", "24"),
        ("--hicache-size", "32"),
        ("--hicache-storage-backend", "nixl"),
        ("--speculative-algorithm", "NEXTN"),
        ("--speculative-num-draft-tokens", "4"),
    ):
        assert argv_after(argv, flag) == value, flag
    assert "--ple-offload-embedding" in argv  # RAM PLE stays on
    assert argv_after(argv, "--chat-template").endswith(
        "configs/pennyroyal/templates/froggeric-v22.5.jinja"
    )
    assert json.loads(argv_after(argv, "--default-chat-template-kwargs")) == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "medium",
    }
    # The NIXL operational config travels from the mounted directory to SGLang.
    assert argv_after(argv, "--hicache-storage-backend-extra-config") == (
        f"@{config / 'nixl-posix.toml'}"
    )
    assert argv_after(argv, "--model-path") == str(tmp_path / "model-share")
    # The writable roots and the derived namespace are the launcher's own.
    assert str(root) in result.stdout  # the runtime really is the image's
    assert (cache / "sglang" / "jit").is_dir()
    namespace = Path(server_env["namespace"])
    assert namespace.is_dir() and str(namespace).startswith(str(nixl))
    assert (namespace / "namespace-identity.json").is_file()
    assert server_env["cache"] == str(cache / "sglang")


def test_mounted_startup_27b_keeps_its_profile(tmp_path):
    config = prepared_config(tmp_path, "start-27b-dflash2.sh")
    result, argv, _, _ = launch(tmp_path, config / "start-27b-dflash2.sh")
    assert result.returncode == 0, result.stderr
    assert argv[0] == "serve"
    for flag, value in (
        ("--speculative-algorithm", "DFLASH"),
        ("--speculative-num-draft-tokens", "8"),
        ("--speculative-draft-window-size", "2048"),
        ("--speculative-draft-kv-cache-dtype", "fp8_e4m3"),
        ("--hicache-size", "96"),
        ("--hicache-storage-backend", "nixl"),
        ("--max-mamba-cache-size", "24"),
        ("--mamba-track-interval", "256"),
        ("--decode-attention-backend", "trtllm_mha"),
    ):
        assert argv_after(argv, flag) == value, flag
    assert argv_after(argv, "--speculative-draft-model-path").endswith("draft")
    assert argv_after(argv, "--hicache-storage-backend-extra-config") == (
        f"@{config / 'nixl-posix.toml'}"
    )


def test_nixl_free_startup_keeps_ram_hicache_and_touches_nothing(tmp_path):
    nixl = tmp_path / "nixl root"
    nixl.mkdir(parents=True, exist_ok=True)
    (nixl / "keep-me" / "bucket").mkdir(parents=True)
    (nixl / "keep-me" / "bucket" / "entry").write_bytes(b"cached")
    config = prepared_config(tmp_path, "start-flash-next-no-nixl.sh")
    result, argv, server_env, _ = launch(
        tmp_path, config / "start-flash-next-no-nixl.sh", keep_nixl_root=True
    )
    assert result.returncode == 0, result.stderr
    assert "--enable-hierarchical-cache" in argv
    assert argv_after(argv, "--hicache-size") == "32"
    assert argv_after(argv, "--hicache-io-backend") == "kernel"
    assert argv_after(argv, "--hicache-mem-layout") == "page_first"
    assert "--hicache-storage-backend" not in argv
    assert not any("nixl" in argument.lower() for argument in argv)
    assert server_env["namespace"] == "unset"
    # No NIXL work means no NIXL root needed and no cache data removed.
    assert (nixl / "keep-me" / "bucket" / "entry").read_bytes() == b"cached"
    assert sorted(path.name for path in nixl.iterdir()) == ["keep-me"]


def test_missing_nixl_config_is_not_replaced_by_an_image_default(tmp_path):
    config = prepared_config(tmp_path, "start-flash-next.sh", with_nixl_config=False)
    result, argv, _, _ = launch(tmp_path, config / "start-flash-next.sh")
    assert result.returncode != 0 and argv is None
    assert "NIXL config missing" in result.stderr
    assert str(config / "nixl-posix.toml") in result.stderr


def test_frspec_startup_keeps_its_pinned_assets(tmp_path):
    # The stand-in checkpoint cannot match the pinned FR-Spec tokenizer, so the
    # mounted recipe must stop rather than serve an unqualified tokenizer.
    config = prepared_config(tmp_path, "start-flash-next-frspec.sh")
    result, argv, _, _ = launch(tmp_path, config / "start-flash-next-frspec.sh")
    assert result.returncode != 0 and argv is None
    assert "Verifying the pinned FR-Spec map" in result.stdout
    assert "tokenizer differs from the qualified FR-Spec tokenizer" in result.stderr


def test_mounted_recipe_launches_exactly_like_the_image_recipe(tmp_path):
    """Same environment in, same server argv out -- for both NIXL profiles."""
    for startup, recipe in (
        ("start-flash-next.sh", RECIPES / "serve-flash-next.sh"),
        ("start-27b-dflash2.sh", RECIPES / "serve-qwen38-27b-dflash2.sh"),
    ):
        config = prepared_config(tmp_path, startup)
        nixl_config = config / "nixl.toml"
        nixl_config.write_text((CONFIGS / "nixl-posix.toml").read_text())
        mounted_result, mounted_argv, _, (root, _, _) = launch(
            tmp_path, config / startup, env={"NIXL_CONFIG": str(nixl_config)}
        )
        recipe_result, recipe_argv, _, _ = launch(
            tmp_path,
            recipe,
            through_entrypoint=False,
            env={"NIXL_CONFIG": str(nixl_config)},
        )
        assert mounted_result.returncode == 0, mounted_result.stderr
        assert recipe_result.returncode == 0, recipe_result.stderr
        assert mounted_argv is not None and recipe_argv is not None, recipe
        assert "--hicache-storage-backend" in recipe_argv
        # The mounted copy reaches the helpers through the stand-in image root;
        # the recipe sees the same files at their repository paths.
        normalised = [
            item.replace(f"{root}/configs/", f"{REPO}/configs/")
            for item in mounted_argv
        ]
        assert normalised == recipe_argv, recipe


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    import inspect
    import tempfile

    for name, case in sorted(globals().items()):
        if name.startswith("test_") and inspect.isfunction(case):
            with tempfile.TemporaryDirectory() as temp:
                case(Path(temp))
            print(f"{name} ok")
