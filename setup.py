import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = os.environ.get("MEANINGFUL_SPOT_BUILD", "minimal").strip().lower()
PYTHON_EXECUTABLE = sys.executable

for stale_path in [
    ROOT / "src" / "build",
    ROOT / "build",
    ROOT / ".conan",
    ROOT / "src" / "meaningful_spot_detector.cpython-313t-x86_64-linux-gnu.so",
    ROOT / "src" / "meaningful_spot_detector.cpython-312-x86_64-linux-gnu.so",
    ROOT / "src" / "meaningful_spot_detector.cpython-311-x86_64-linux-gnu.so",
]:
    if stale_path.exists():
        if stale_path.is_dir():
            shutil.rmtree(stale_path, ignore_errors=True)
        else:
            stale_path.unlink(missing_ok=True)

conan_home = Path(os.environ.get("CONAN_HOME", str(ROOT / ".conan")))
conan_home.mkdir(parents=True, exist_ok=True)
os.environ["CONAN_HOME"] = str(conan_home)

conan_dir = Path(os.environ.get("MEANINGFULSPOT_CONAN_OUTPUT_DIR", str(ROOT / "build" / "conan")))
conan_dir.mkdir(parents=True, exist_ok=True)
os.environ["MEANINGFULSPOT_CONAN_OUTPUT_DIR"] = str(conan_dir)

if TARGET in {"minimal", "no_maxtree", "lightweight"}:
    SCRIPT = ROOT / "src" / "setup_no_maxtree.py"
elif TARGET in {"full", "maxtree", "max_tree", "max-tree"}:
    SCRIPT = ROOT / "src" / "setup.py"
else:
    raise ValueError(
        "MEANINGFUL_SPOT_BUILD must be one of: minimal, no_maxtree, lightweight, "
        "full, maxtree, max_tree, max-tree"
    )

if not SCRIPT.exists():
    raise FileNotFoundError(f"Build script not found: {SCRIPT}")

if TARGET in {"full", "maxtree", "max_tree", "max-tree"}:
    conan_home = Path(os.environ.get("CONAN_HOME", str(ROOT / ".conan")))
    conan_home.mkdir(parents=True, exist_ok=True)
    conan_dir = Path(os.environ.get("MEANINGFULSPOT_CONAN_OUTPUT_DIR", str(ROOT / "build" / "conan")))
    conan_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CONAN_HOME"] = str(conan_home)
    os.environ["MEANINGFULSPOT_CONAN_OUTPUT_DIR"] = str(conan_dir)
    env = os.environ.copy()
    env["CONAN_HOME"] = str(conan_home)
    env["MEANINGFULSPOT_CONAN_OUTPUT_DIR"] = str(conan_dir)
    env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"

    def get_conan_cmd() -> list[str] | None:
        # Note: the `conan` package intentionally ships no `__main__.py`, so
        # `python -m conan` never works. The `conan` console-script entry
        # point (installed alongside the package) is the only valid CLI
        # invocation, so it must come first here.
        candidates = [
            [shutil.which("conan")] if shutil.which("conan") else None,
            [sys.executable, "-m", "conan"],
        ]
        candidates = [cmd for cmd in candidates if cmd is not None]
        for cmd in candidates:
            try:
                res = subprocess.run(
                    cmd + ["--version"],
                    check=True,
                    env=env,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
            first_line = (res.stdout or "").splitlines()[0] if (res.stdout or "") else ""
            version_token = first_line.split()[-1] if first_line else ""
            try:
                major = int(version_token.split(".")[0])
            except (IndexError, ValueError):
                continue
            if major >= 2:
                return cmd

        return None

    conan_cmd = get_conan_cmd()
    if conan_cmd is None:
        # Fall back to installing Conan into the active interpreter's
        # environment, then retry discovery.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "conan>=2.0"],
            check=True,
            env=env,
        )
        conan_cmd = get_conan_cmd()
    if conan_cmd is not None:
        conan_args = []

        profile_path = conan_home / "profiles" / "default"
        profile_is_incomplete = True
        if profile_path.exists():
            profile_text = profile_path.read_text(encoding="utf-8")
            profile_is_incomplete = "compiler=" not in profile_text
        if not profile_path.exists() or profile_is_incomplete:
            subprocess.run(
                [*conan_cmd, *conan_args, "profile", "detect", "--force"],
                check=True,
                env=env,
            )
        subprocess.run(
            [
                *conan_cmd,
                *conan_args,
                "install",
                str(ROOT),
                "--output-folder",
                str(conan_dir),
                "--build=missing",
                # Conan Center's prebuilt 'b2' tool binary (used to bootstrap
                # Boost's own build) is linked against a newer glibc than
                # manylinux images provide, causing 'GLIBC_x.xx not found' at
                # runtime. Force it to be rebuilt from source instead.
                "--build=b2/*",
                # Build all dependencies (boost, tbb, fmt, hwloc, bzip2, ...)
                # statically so the produced extension has no runtime shared
                # library dependencies. Otherwise auditwheel/delocate can't
                # locate libtbb.so.12 / libtbb.12.dylib (they live in Conan's
                # cache, not a system library path) and repairing the wheel
                # fails.
                "-o",
                "*:shared=False",
                # onetbb's "tbbbind" feature (NUMA-aware thread binding, not
                # needed here) unconditionally requires hwloc built shared,
                # which conflicts with the blanket static override above.
                # Disable it so hwloc stays static too.
                "-o",
                "onetbb/*:tbbbind=False",
                "-s",
                "build_type=Release",
                "-s:a",
                "compiler.cppstd=20",
            ],
            check=True,
            env=env,
        )
    else:
        raise RuntimeError(
            "MEANINGFUL_SPOT_BUILD=full selected, but Conan 2.x is not available in this build environment. "
            "Install Conan 2.x before building the full Max-Tree variant."
        )

sys.path.insert(0, str(SCRIPT.parent))
runpy.run_path(str(SCRIPT), run_name="__main__")
