"""
This file enables you to build both the version
of the code that uses Max-Trees and the version
that does use a local level set search.
"""

from Cython.Build import cythonize
import distutils
import distutils.ccompiler
import glob
import os
import platform
import re
import shutil
from pathlib import Path
import numpy as np
from setuptools import setup
from setuptools.extension import Extension

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
os.chdir(SRC_DIR)
README = ROOT / "README.md"
ROOT_EXTERNALS = ROOT / "externals"
X86_MACHINES = {"x86_64", "AMD64", "i386", "i686"}


def _stage_license_files() -> list[str]:
    """Copy the license files into src/_licenses/ so setuptools' license_files
    globs (which must stay within the package root once cwd is SRC_DIR) can
    find them without escaping upward via '..'."""
    license_dir = SRC_DIR / "_licenses"
    sources = {
        "LICENSE": ROOT / "LICENSE",
        "THIRD_PARTY_NOTICES.md": ROOT / "THIRD_PARTY_NOTICES.md",
        "externals/pylene/LICENSE": ROOT / "externals" / "pylene" / "LICENSE",
        "externals/xsimd/LICENSE": ROOT / "externals" / "xsimd" / "LICENSE",
    }
    staged = []
    for rel_dest, src in sources.items():
        dest = license_dir / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        # setuptools' license-file glob patterns require '/' regardless of OS.
        staged.append(dest.relative_to(SRC_DIR).as_posix())
    return staged


def _conan_include_and_lib_dirs() -> tuple[list[str], list[str]]:
    includedirs: set[str] = set()
    libdirs: set[str] = set()

    def add_from_env_script(path: Path):
        if not path.exists():
            return
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in ("CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH", "LD_LIBRARY_PATH"):
            match = re.search(rf"export\s+{re.escape(pattern)}=(.*)", text)
            if not match:
                continue
            value = match.group(1).strip().strip('"').strip("'")
            for item in value.split(os.pathsep):
                if not item:
                    continue
                # Conan's generated env scripts sometimes reference the
                # previous value of the variable using bash parameter
                # expansion, e.g. '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}'.
                # A naive ':' split breaks that syntax into bogus tokens
                # (e.g. '${LD_LIBRARY_PATH', '+', '$LD_LIBRARY_PATH}'), so
                # skip anything that isn't a plain absolute filesystem path.
                if any(ch in item for ch in ("$", "{", "}", "`")):
                    continue
                if not os.path.isabs(item):
                    continue
                if pattern in {"CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"}:
                    includedirs.add(item)
                else:
                    libdirs.add(item)

    extra_output_dir = Path(os.environ.get("MEANINGFULSPOT_CONAN_OUTPUT_DIR", ""))
    search_roots = [
        ROOT / "build" / "conan",
        ROOT / ".conan",
        Path(os.environ.get("CONAN_HOME", ROOT / ".conan")),
        extra_output_dir if str(extra_output_dir) else None,
        Path.home() / ".conan",
        Path.home() / ".conan2",
    ]
    search_roots = [root for root in search_roots if root is not None]
    for root in search_roots:
        if not root.exists():
            continue
        for script in sorted(root.glob("**/conanbuildenv-*.sh")):
            add_from_env_script(script)
        for script in sorted(root.glob("**/conanrunenv-*.sh")):
            add_from_env_script(script)

    extra_output_dir = Path(os.environ.get("MEANINGFULSPOT_CONAN_OUTPUT_DIR", ""))
    for candidate in (
        ROOT / "build" / "conan",
        ROOT / ".conan",
        Path(os.environ.get("CONAN_HOME", ROOT / ".conan")),
        extra_output_dir if str(extra_output_dir) else None,
        Path.home() / ".conan" / "p",
        Path.home() / ".conan2" / "p",
    ):
        if candidate is None:
            continue
        if not candidate.exists():
            continue
        for include_dir in candidate.glob("**/include"):
            if include_dir.is_dir():
                includedirs.add(str(include_dir))
                eigen_dir = include_dir / "eigen3"
                if eigen_dir.is_dir():
                    includedirs.add(str(eigen_dir))
        for lib_name in ("lib", "lib64"):
            for lib_dir in candidate.glob(f"**/{lib_name}"):
                if lib_dir.is_dir():
                    libdirs.add(str(lib_dir))

    return sorted(includedirs), sorted(libdirs)

def compilerName() -> str:
    """Return the name of the compiler."""
    compiler = distutils.ccompiler.get_default_compiler()
    return compiler

xsimd_dir = str(ROOT_EXTERNALS / "xsimd" / "include")
pylene_dir = str(ROOT_EXTERNALS / "pylene" / "pylene" / "include")

if compilerName() == "msvc":
    # /utf-8 is required by fmt (static assertion on Unicode support) and is
    # generally correct for source files that may contain non-ASCII text.
    cc_args = ["/O2", "/std:c++20", "/favor:INTEL64", "/MACHINE:X64", "/utf-8"]
    if platform.machine() in X86_MACHINES:
        cc_args.insert(1, "/arch:AVX2")
    ll_args = []
    additional_include_dirs = []
else:
    # Pylene needs rangev3, eigen3 and boost. These are resolved automatically by Conan in the
    # full build path, and we fall back to system headers on non-Conan builds.
    cc_args = ["-O3", "-std=c++20"]
    if platform.machine() in X86_MACHINES:
        cc_args.extend(["-mavx", "-mavx2", "-mfma"])
    if platform.system() == "Linux" and platform.machine() in X86_MACHINES:
        cc_args.append("-mtls-dialect=gnu2")
    ll_args = cc_args
    additional_include_dirs = []
    if os.path.isdir("/usr/include/eigen3"):
        additional_include_dirs.append("/usr/include/eigen3/")

conan_includes, conan_libs = _conan_include_and_lib_dirs()
if not conan_includes or not any(Path(d).name == "eigen3" for d in conan_includes):
    conan_home = Path(os.environ.get("CONAN_HOME", str(ROOT / ".conan")))
    conan_output_dir = Path(os.environ.get("MEANINGFULSPOT_CONAN_OUTPUT_DIR", str(ROOT / "build" / "conan")))
    raise RuntimeError(
        "The full (Max-Tree) build requires Conan-installed dependencies "
        "(boost, eigen, fmt, hwloc, onetbb, range-v3), but no usable include "
        f"directories were found.\nSearched:\n  CONAN_HOME={conan_home}\n"
        f"  MEANINGFULSPOT_CONAN_OUTPUT_DIR={conan_output_dir}\n"
        f"  {ROOT / 'build' / 'conan'}\n  {ROOT / '.conan'}\n"
        f"  {Path.home() / '.conan2'}\n"
        "Run 'conan install . --output-folder=build/conan --build=missing "
        "-s build_type=Release -s:a compiler.cppstd=20' from the repo root first, "
        "or build via the root setup.py which does this automatically."
    )
additional_include_dirs = conan_includes + additional_include_dirs
os.environ["CPLUS_INCLUDE_PATH"] = os.pathsep.join(conan_includes + [os.environ.get("CPLUS_INCLUDE_PATH", "")])
os.environ["C_INCLUDE_PATH"] = os.pathsep.join(conan_includes + [os.environ.get("C_INCLUDE_PATH", "")])

if compilerName() == "msvc":
    # Windows/MSVC: add library directories and link TBB/fmt
    if conan_libs:
        ll_args = [f"/LIBPATH:{path}" for path in conan_libs] + ["tbb12.lib", "fmt.lib"]
        os.environ["LIBRARY_PATH"] = os.pathsep.join(conan_libs + [os.environ.get("LIBRARY_PATH", "")])
else:
    # Linux/macOS: use GCC-style linking with TBB/fmt in a linker group
    static_group_libs = ["-ltbb", "-lfmt"]
    if platform.system() == "Linux":
        # --start-group/--end-group are GNU ld options; macOS's linker
        # (ld64/lld) doesn't understand them.
        static_group = ["-Wl,--start-group"] + static_group_libs + ["-Wl,--end-group"]
    else:
        static_group = static_group_libs
    if conan_libs:
        ll_args = cc_args + [f"-L{path}" for path in conan_libs] + static_group
        os.environ["LIBRARY_PATH"] = os.pathsep.join(conan_libs + [os.environ.get("LIBRARY_PATH", "")])
    else:
        ll_args = cc_args + static_group

pylene_cpp_dir = str(ROOT_EXTERNALS / "pylene" / "pylene" / "src")
all_pylene_cpp_files = []
if platform.machine() in X86_MACHINES:
    all_pylene_cpp_files = glob.glob("**/*.cpp", root_dir=pylene_cpp_dir, recursive=True)
    # glob() returns OS-native separators (backslashes on Windows), so the
    # exclusion below must normalize before comparing against "io/".
    all_pylene_cpp_files = [f for f in all_pylene_cpp_files if not f.replace(os.sep, "/").startswith("io/")]
    all_pylene_cpp_files = [os.path.join(pylene_cpp_dir, f) for f in all_pylene_cpp_files]

extensions = [
    Extension(
        "meaningful_spot_detector",
        [
            "spot_detector_cpp.pyx",
            "maxpool.cpp",
            "meaningful_ll.cpp",
            "spot_detector.cpp",
            "spot_detector_maxtree.cpp",
        ] + all_pylene_cpp_files,
        language="c++",
        include_dirs=additional_include_dirs + [np.get_include(), xsimd_dir, pylene_dir],
        extra_compile_args=cc_args,
        extra_link_args=ll_args
    )
]

setup(
    name="meaningful-spot-detector",
    version="1.0.0",
    description="Meaningful spot detector with the local level-set implementation and the full Max-Tree variant.",
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url='https://github.com/axeldavy/MeaningfulSpot',
    license='MIT',
    license_files=_stage_license_files(),
    python_requires='>=3.10',
    install_requires=['numpy>=1.26'],
    package_data={'': ['*.pyi']},
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3", 'freethreading_compatible': True})
)
