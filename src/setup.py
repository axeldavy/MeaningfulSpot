"""
This file enables you to build both the version
of the code that uses Max-Trees and the version
that does use a local level set search.
"""

from Cython.Build import cythonize
import distutils
import distutils.ccompiler
import glob
import numpy as np
import os
from setuptools import setup
from setuptools.extension import Extension

def compilerName() -> str:
    """Return the name of the compiler."""
    compiler = distutils.ccompiler.get_default_compiler()
    return compiler

xsimd_dir = os.path.join("..", "externals", "xsimd", "include")
pylene_dir = os.path.join("..", "externals", "pylene", "pylene", "include")

if compilerName() == "msvc":
    cc_args = ["/O2", "/arch:AVX2", "/std:c++20", "/favor:INTEL64", "/MACHINE:X64"]
    # Pylene requires FreeImage, rangev3 and boost. On Windows, we assume they
    # are not available and you should put them in the externals directory.
    ll_args = [
        os.path.join("..", "externals", "FreeImage", "FreeImage.lib")
    ]
    additional_include_dirs = [
        os.path.join("..", "externals", "FreeImage"),
        os.path.join("..", "externals", "boost", "include"),
        os.path.join("..", "externals", "range-v3", "include")
    ]
else:
    # Since we use thread_local variables in the hot path, the default linux gnu tls behaviour
    # it expensive (initialize on first use). gnu2 gives a significant performance boost.
    # Pylene requires FreeImage, rangev3, eigen3 and boost. On Linux, we assume they are available
    # as system libraries.
    cc_args = ["-mavx", "-mavx2", "-O3", "-std=c++20", "-mfma", "-march=native",
               "-mtls-dialect=gnu2", "-ltbb", "-lfreeimage", '-lfmt', '-lcfitsio']
    ll_args = cc_args
    additional_include_dirs = ["/usr/include/eigen3/"]

pylene_cpp_dir = os.path.join("..", "externals", "pylene", "pylene", "src")
all_pylene_cpp_files = glob.glob("**/*.cpp", root_dir=pylene_cpp_dir, recursive=True)
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
    name="meaningful_spot_detector",
    version="1.0.0",
    url='https://github.com/axeldavy/MeaningfulSpot',
    license='MIT AND MPL-2.0',
    python_requires='>=3.10',
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3", 'freethreading_compatible': True})
)
