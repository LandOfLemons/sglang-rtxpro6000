"""CPU-only packaging checks for the accepted FlashInfer SM120 source.

No GPU, no CUDA toolchain, no container runtime and no wheel download: the
packaging rules are exercised against synthetic installed-source trees, and the
carried mailboxes are checked against their pins and attribution. Compilation,
image build and GPU behaviour are qualified separately on the RTX PRO 6000 host;
nothing here claims them.
"""

import hashlib
import importlib.util
import inspect
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

REPO = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO / "scripts" / "pennyroyal" / "flashinfer"
INSTALLER = PACKAGE_DIR / "install.py"
MANIFEST = PACKAGE_DIR / "accepted-sources.json"
DOCKERFILE = REPO / "docker" / "pennyroyal" / "Dockerfile"
CHECK = REPO / "docker/pennyroyal/check_install.py"
BUILD_GUIDE = REPO / "BUILD.md"
NEXT_SCRIPTS = (
    REPO / "configs/pennyroyal/serve-flash-next.sh",
    REPO / "configs/pennyroyal/serve-flash-next-frspec.sh",
    REPO / "docker/pennyroyal/launch/config/start-flash-next.sh",
    REPO / "docker/pennyroyal/launch/config/start-flash-next-frspec.sh",
)
OTHER_SCRIPTS = (
    REPO / "configs/pennyroyal/serve-qwen38-27b-dflash2.sh",
    REPO / "docker/pennyroyal/launch/config/start-27b-dflash2.sh",
)
GDN_EXPORT = (
    'export FLASHINFER_GDN_FP16_ACCUM_MMA="${FLASHINFER_GDN_FP16_ACCUM_MMA:-1}"'
)
DIGEST = re.compile(r"\A[0-9a-f]{64}\Z")


def installer():
    spec = importlib.util.spec_from_file_location("penny_flashinfer_sm120", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def installed_tree(root: Path, contents: dict, version: str) -> Path:
    """A stand-in site-packages: the package, its metadata, the given sources."""
    package = root / "flashinfer"
    for name, content in contents.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    info = root / f"flashinfer_python-{version}.dist-info"
    info.mkdir(parents=True, exist_ok=True)
    (info / "METADATA").write_text(f"Name: flashinfer-python\nVersion: {version}\n")
    return package


def synthetic_source(root: Path) -> dict:
    """A generated mailbox over a generated file, pinned the accepted way.

    The carried mailboxes cover six real FlashInfer files; every packaging rule
    below is per file, so a two-line stand-in exercises the stock, accepted and
    unexpected states without a wheel download.
    """
    patch = root / "generated-source.patch"
    patch.write_text(
        "From 0000000000000000000000000000000000000001 Mon Sep 17 00:00:00 2001\n"
        "From: Test <test@example.com>\n"
        "Subject: [PATCH] test: the accepted generated source\n"
        "\n"
        "---\n"
        "diff --git a/kernel.py b/kernel.py\n"
        "--- a/kernel.py\n"
        "+++ b/kernel.py\n"
        "@@ -1,2 +1,2 @@\n"
        " # stock kernel\n"
        "-return 1\n"
        "+return 2\n"
        "-- \n"
        "2.56.0\n"
    )
    return {
        "flashinfer_python": "9.9.9",
        "flashinfer_jit_cache": "9.9.9+cu130",
        "cuda_arch_list": "12.0f",
        "aot_module": "fused_moe_120",
        "aot_path": "data/aot/fused_moe_120/fused_moe_120.so",
        "patches": [
            {
                "patch": str(patch),
                "root": ".",
                "author": "Test <test@example.com>",
                "commits": ["0" * 40],
                "files": [
                    {
                        "path": "kernel.py",
                        "stock_sha256": digest("# stock kernel\nreturn 1\n"),
                        "accepted_sha256": digest("# stock kernel\nreturn 2\n"),
                    }
                ],
            }
        ],
    }


def raises(call, needle: str) -> None:
    try:
        call()
    except RuntimeError as error:
        assert needle in str(error), f"expected {needle!r} in {error}"
    else:
        raise AssertionError(f"nothing failed; expected {needle!r}")


def test_carried_mailboxes_are_the_accepted_source():
    source = source_manifest()
    assert source["flashinfer_python"] == "0.7.0.post1"
    assert source["flashinfer_jit_cache"] == "0.7.0.post1+cu130"
    assert source["cuda_arch_list"] == "12.0f"
    assert source["aot_path"] == "data/aot/fused_moe_120/fused_moe_120.so"
    assert (
        source["flashinfer_base_commit"] == "946200de1ae94fc93fdd0926f0a13afd1fa7f0f1"
    )
    expected = {
        "patches/moe-source.patch": {
            "commits": [
                "5e86c489f5759cb4006d3b1b5e7bdd14f7a581df",
                "b9fa8893102dc3bbcec92762230e7f1678644a4e",
                "2a4d8d3a9501bf3b3fe3b78d7c6bad38bfc76064",
            ],
            "paths": [
                "csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh",
                "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h",
            ],
            "author": "Penny <Pennyroyal@agentmail.to>",
            "root": "data",
            # The donor lineage the accepted commits record.
            "lineage": [
                "aiueo52/flash-next-rtxpro6000",
                "524af49abcca66fcb4377ba8297022804535fccf",
            ],
        },
        "patches/gdn-source.patch": {
            "commits": ["0b0ba4c2b18173303b46dd8ec381735e1615b313"],
            "paths": [
                "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_cp_sm120.py",
                "flashinfer/gdn_kernels/delta_rule_dsl/delta_rule_sm120.py",
                "flashinfer/gdn_kernels/delta_rule_dsl/helpers.py",
                "flashinfer/gdn_prefill.py",
            ],
            "author": "aa24aa <2496788660@qq.com>",
            "root": "..",
            # Upstream FlashInfer #6227, cherry picked from the accepted commit.
            "lineage": ["c0771c79b7e2f2bc0edf4fdcb7c43b986a56707a", "#6227"],
        },
    }
    by_name = {entry["patch"]: entry for entry in source["patches"]}
    assert set(by_name) == set(expected)
    for name, want in expected.items():
        entry = by_name[name]
        mailbox = (PACKAGE_DIR / name).read_text()
        assert entry["commits"] == want["commits"], name
        assert entry["author"] == want["author"], name
        assert entry["root"] == want["root"], name
        assert sorted(spec["path"] for spec in entry["files"]) == sorted(
            want["paths"]
        ), name
        assert f"From: {want['author']}" in mailbox, name
        for commit in want["commits"]:
            assert mailbox.count(f"From {commit} ") == 1, (name, commit)
        for token in want["lineage"]:
            # Donor and upstream ids live in the mailbox or in the manifest's
            # attribution line, never in a comment someone can drop silently.
            assert token in mailbox or token in entry["attribution"], (name, token)
        assert want["commits"] == entry["commits"], name
        for spec in entry["files"]:
            assert DIGEST.match(spec["stock_sha256"]), spec["path"]
            assert DIGEST.match(spec["accepted_sha256"]), spec["path"]
            assert spec["stock_sha256"] != spec["accepted_sha256"], spec["path"]
    # Scope guards: the accepted input is the six production files, nothing else.
    assert len(by_name["patches/moe-source.patch"]["files"]) == 2
    assert len(by_name["patches/gdn-source.patch"]["files"]) == 4
    assert (
        "FLASHINFER_MOE_FUSED_PROLOGUE"
        in (PACKAGE_DIR / "patches/moe-source.patch").read_text()
    )
    assert (
        "FLASHINFER_GDN_FP16_ACCUM_MMA"
        in (PACKAGE_DIR / "patches/gdn-source.patch").read_text()
    )


def test_accepted_pin_matches_every_dependency_pin():
    source = source_manifest()
    version, cache = source["flashinfer_python"], source["flashinfer_jit_cache"]
    assert cache == f"{version}+cu130"
    assert (
        f'"flashinfer_python[cu13]=={version}"'
        in (REPO / "python/pyproject.toml").read_text()
    )
    dockerfile = DOCKERFILE.read_text()
    assert f"'flashinfer-jit-cache=={version}+cu130'" in dockerfile
    assert f"'flashinfer-jit-cache-sm120f=={version}+cu130'" in dockerfile
    assert f'"flashinfer-python": "{version}"' in CHECK.read_text()


def test_stock_source_is_patched_once_and_a_repeat_run_accepts_it(tmp_path):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 1\n"}, "9.9.9"
    )
    patch = Path(source["patches"][0]["patch"])
    # A Pennyroyal checkout around the environment must not redirect git apply.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    assert module.apply_accepted_source(package, source) == [f"{patch}: applied"]
    assert (package / "kernel.py").read_text() == "# stock kernel\nreturn 2\n"
    # The same command again is a no-op, not an error and not a double patch.
    assert module.apply_accepted_source(package, source) == [
        f"{patch}: already patched"
    ]
    assert (package / "kernel.py").read_text() == "# stock kernel\nreturn 2\n"


def test_unexpected_source_and_wrong_pin_fail_before_building(tmp_path):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# mine\nreturn 3\n"}, "9.9.9"
    )
    raises(
        lambda: module.apply_accepted_source(package, source),
        "neither the stock nor the accepted",
    )

    # A partly applied state is a different answer from a repeat installation.
    source["patches"][0]["files"].append(
        {
            "path": "second.py",
            "stock_sha256": digest("second\n"),
            "accepted_sha256": digest("patched\n"),
        }
    )
    (package / "kernel.py").write_text("# stock kernel\nreturn 1\n")
    (package / "second.py").write_text("changed\n")
    raises(
        lambda: module.apply_accepted_source(package, source),
        "neither the stock nor the accepted",
    )

    for version in ("0.6.17", "0.7.1"):
        other = installed_tree(tmp_path / f"site-{version}", {"kernel.py": ""}, version)
        raises(
            lambda other=other: module.require_accepted_pin(source, other),
            "not the accepted 9.9.9 source pin",
        )
    missing = tmp_path / "site-nodist" / "flashinfer"
    missing.mkdir(parents=True)
    raises(
        lambda: module.require_accepted_pin(source, missing),
        "no flashinfer_python .dist-info",
    )


def test_installed_check_needs_the_accepted_source_and_the_package_local_module(
    tmp_path,
):
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 1\n"}, "9.9.9"
    )
    stock_kernel = tmp_path / "provider" / "fused_moe_120.so"
    stock_kernel.parent.mkdir()
    stock_kernel.write_bytes(b"stock provider bytes")
    installed = package / source["aot_path"]
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"stock provider bytes")

    # Stock source with the provider wheel: what an ordinary 0.7.0.post1 install
    # looks like, and what used to satisfy the image check.
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "is stock FlashInfer source",
    )
    module.apply_accepted_source(package, source)
    # Patched source, but the preferred path only holds a copy of the prebuilt
    # provider kernel: nothing was built from the accepted source.
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "copy of the stock provider kernel",
    )
    installed.unlink()
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "absent or empty",
    )

    installed.write_bytes(b"built from patched bytes")
    info = module.check_installed(package, source, [stock_kernel])
    assert info["sm120_module"] == str(installed)
    assert info["sm120_module_bytes"] == len(b"built from patched bytes")
    assert (
        info["sm120_module_sha256"]
        == hashlib.sha256(b"built from patched bytes").hexdigest()
    )
    assert info["flashinfer"] == "9.9.9"
    assert info["accepted_source"] == [
        {
            "patch": str(tmp_path / "generated-source.patch"),
            "author": "Test <test@example.com>",
            "commits": ["0" * 40],
        }
    ]

    # Copying the stock prebuilt kernel into the preferred path is not a build,
    # and without a provider wheel to compare against the module is still required.
    installed.write_bytes(stock_kernel.read_bytes())
    raises(
        lambda: module.check_installed(package, source, [stock_kernel]),
        "copy of the stock provider kernel",
    )
    module.check_installed(package, source)


def test_the_image_check_refuses_an_unpackaged_flashinfer(tmp_path, monkeypatch):
    """docker/pennyroyal/check_install.py must not pass on a stock install."""
    installed_tree(tmp_path / "site", {"gdn_prefill.py": "# stock\n"}, "0.7.0.post1")
    spec = importlib.util.spec_from_file_location("penny_container_check", CHECK)
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    importlib.invalidate_caches()
    try:
        check.check_flashinfer_sm120(REPO, [])
    except RuntimeError as error:
        assert "FlashInfer source" in str(error), str(error)
    else:
        raise AssertionError("the container check passed on a stock FlashInfer")


def test_build_targets_the_patched_sources_without_the_prebuilt_shortcut():
    code = installer().BUILD_CODE
    assert "gen_cutlass_fused_moe_sm120_module(use_fast_build=False)" in code
    assert "spec.build()" in code
    assert "print(pathlib.Path(spec.jit_library_path))" in code
    # The shortcuts that take the stock AOT module and compile nothing.
    for shortcut in ("build_and_load", "build_jit_specs", "skip_prebuilt"):
        assert shortcut not in code, shortcut
    # The patched headers of the selected package, not another source tree.
    assert "PENNY_FLASHINFER_ACCEPTED_CSRC" in code
    assert "FLASHINFER_CSRC_DIR" in code
    text = INSTALLER.read_text()
    assert 'FLASHINFER_CUDA_ARCH_LIST=source["cuda_arch_list"]' in text
    assert source_manifest()["cuda_arch_list"] == "12.0f"
    for host_path in ("/home/", "thegrid", ".aot-overlay"):
        assert host_path not in text, host_path


def test_both_installation_paths_run_the_one_step():
    guide = BUILD_GUIDE.read_text()
    dockerfile = DOCKERFILE.read_text()
    fresh = guide.split("## Fresh install", 1)[1].split(
        "## Update an existing install", 1
    )[0]
    update = guide.split("## Update an existing install", 1)[1].split(
        "## NIXL POSIX", 1
    )[0]
    assert "scripts/pennyroyal/flashinfer/install.py" in fresh
    assert "scripts/pennyroyal/flashinfer/install.py" in update
    assert "--no-deps -e python" in update
    assert "flashinfer-python" in update  # the prerequisite is spelled out
    jit_cache = dockerfile.index("flashinfer-jit-cache-sm120f")
    packaging = dockerfile.index("scripts/pennyroyal/flashinfer/install.py")
    freeze = dockerfile.index("pip check")
    assert jit_cache < packaging < freeze
    assert "PENNY_BUILD_JOBS=4" in dockerfile and "MAX_JOBS=4" in dockerfile
    assert "check_flashinfer_sm120" in CHECK.read_text()


def test_a_stale_jit_cache_shim_stops_the_upgrade_before_the_build(tmp_path):
    """The 0.6.17 -> 0.7.0.post1 native sequence, checked on a stand-in site.

    flashinfer-python's own metadata does not pull the JIT-cache family, so an
    upgraded environment keeps the old shim, and FlashInfer aborts its import
    over the mismatch. The step has to name that package instead of dying inside
    the compile.
    """
    module = installer()
    source = synthetic_source(tmp_path)
    package = installed_tree(
        tmp_path / "site", {"kernel.py": "# stock kernel\nreturn 2\n"}, "9.9.9"
    )
    (package / source["aot_path"]).parent.mkdir(parents=True)
    (package / source["aot_path"]).write_bytes(b"built")
    site = tmp_path / "site"

    def dist_info(name, version):
        info = site / f"{name}-{version}.dist-info"
        info.mkdir(exist_ok=True)
        (info / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")

    # No cache family at all: genuinely optional, the step still accepts it.
    module.check_installed(package, source)

    # The stale shim a v2.5.x environment keeps beside the new FlashInfer.
    dist_info("flashinfer_jit_cache", "0.6.17+cu130")
    dist_info("flashinfer_jit_cache_sm120f", "0.6.17+cu130")
    for call in (
        lambda: module.check_installed(package, source),
        lambda: module.install_accepted_source(package, source),
    ):
        raises(call, "flashinfer-jit-cache 0.6.17+cu130")
    # The documented remedy: the same pinned family the image installs, which
    # replaces the old wheel. The provider may stay old -- FlashInfer skips an
    # incompatible provider with a warning -- because only the shim breaks the
    # import, and only the shim is refused here.
    shutil.rmtree(site / "flashinfer_jit_cache-0.6.17+cu130.dist-info")
    dist_info("flashinfer_jit_cache", "9.9.9+cu130")
    module.check_installed(package, source)


def test_the_documented_recovery_aligns_the_whole_flashinfer_family():
    guide = BUILD_GUIDE.read_text()
    # The one fenced command block that touches the JIT-cache family: the prose
    # may move, the block is what an operator pastes.
    blocks = [part for part in guide.split("```") if "flashinfer-jit-cache==" in part]
    assert len(blocks) == 1, blocks
    recovery = blocks[0]
    source = source_manifest()
    for spec in (
        f"'flashinfer-python[cu13]=={source['flashinfer_python']}'",
        f"'flashinfer-jit-cache=={source['flashinfer_jit_cache']}'",
        f"'flashinfer-jit-cache-sm120f=={source['flashinfer_jit_cache']}'",
    ):
        assert spec in recovery, spec
    assert "--no-deps --index-url https://flashinfer.ai/whl/cu130" in recovery
    assert "scripts/pennyroyal/flashinfer/install.py" in recovery
    # Both source selections name a release ref rather than an older tag, so the
    # step cannot be documented against a checkout that lacks it.
    for sequence in (
        guide.split("## Fresh install", 1)[1].split("## Update an existing install", 1)[
            0
        ],
        guide.split("## Update an existing install", 1)[1].split("## NIXL POSIX", 1)[0],
    ):
        assert "$RELEASE_REF" in sequence, sequence[:200]
        assert "scripts/pennyroyal/flashinfer/install.py" in sequence
    assert "--branch pennyroyal-v" not in guide
    assert "switch --detach pennyroyal-v" not in guide


def test_next_recipes_default_the_accepted_gdn_mode_and_27b_does_not():
    for script in NEXT_SCRIPTS:
        lines = script.read_text().splitlines()
        assert GDN_EXPORT in lines, script
        # Before the interpreter or the server starts, so FlashInfer reads it.
        first_start = min(
            index
            for index, line in enumerate(lines)
            if '"$PYTHON"' in line or '"$SGLANG_EXE"' in line
        )
        assert lines.index(GDN_EXPORT) < first_start, script
    for script in OTHER_SCRIPTS:
        assert "FLASHINFER_GDN_FP16_ACCUM_MMA" not in script.read_text(), script
    # The accepted source keeps the mode off; no global FlashInfer default moved.
    assert (
        "FLASHINFER_GDN_FP16_ACCUM_MMA"
        not in (REPO / "python/pyproject.toml").read_text()
    )
    entrypoint = (REPO / "docker/pennyroyal/entrypoint.sh").read_text()
    assert "FLASHINFER_GDN_FP16_ACCUM_MMA" not in entrypoint


if __name__ == "__main__":  # pytest is the real driver; this is a smoke check
    for name, case in sorted(globals().items()):
        if not (name.startswith("test_") and inspect.isfunction(case)):
            continue
        with tempfile.TemporaryDirectory() as temp:
            case(Path(temp)) if inspect.signature(case).parameters else case()
        print("pass", name)
