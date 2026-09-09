"""
This file enables you to build only the version
of the code that does not support maxtrees.

Calls to the maxtree implementation will fallback
to the non-maxtree one.

This enables you to test the non-maxtree code (no
dependencies), when meeting the dependencies of the
maxtree code are challenging.
"""

import distutils
import distutils.ccompiler
import platform
import shutil
from pathlib import Path
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
os.chdir(SRC_DIR)
README = ROOT / "README.md"
ROOT_EXTERNALS = ROOT / "externals"


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


def compilerName() -> str:
    """Return the name of the compiler."""
    compiler = distutils.ccompiler.get_default_compiler()
    return compiler

xsimd_dir = str(ROOT_EXTERNALS / "xsimd" / "include")

if compilerName() == "msvc":
    cc_args = ["/O2", "/arch:AVX2", "/std:c++20", "/favor:INTEL64", "/MACHINE:X64", "/utf-8"]
    ll_args = []
else:
    cc_args = ["-O3", "-std=c++20"]
    if platform.machine() in {"x86_64", "AMD64", "i386", "i686"}:
        cc_args.extend(["-mavx", "-mavx2", "-mfma", "-march=native"])
    if platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64", "i386", "i686"}:
        cc_args.append("-mtls-dialect=gnu2")
    ll_args = cc_args

extensions = [
    Extension(
        "meaningful_spot_detector",
        [
            "spot_detector_cpp.pyx",
            "maxpool.cpp",
            "meaningful_ll.cpp",
            "spot_detector.cpp",
            "spot_detector_maxtree_stub.cpp",
        ],
        language="c++",
        include_dirs=[np.get_include(), xsimd_dir],
        extra_compile_args=cc_args,
        extra_link_args=ll_args
    )
]

setup(
    name="meaningful-spot-detector",
    version="1.0.0",
    description="Meaningful spot detector using local level sets and an optional Max-Tree implementation.",
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