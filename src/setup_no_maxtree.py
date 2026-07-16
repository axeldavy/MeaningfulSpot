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
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
import os

def compilerName() -> str:
    """Return the name of the compiler."""
    compiler = distutils.ccompiler.get_default_compiler()
    return compiler

xsimd_dir = os.path.join("external", "xsimd", "include")

if compilerName() == "msvc":
    cc_args = ["/O2", "/arch:AVX2", "/std:c++20", "/favor:INTEL64", "/MACHINE:X64"]
    ll_args = []
else:
    # Since we use thread_local variables in the hot path, the default linux gnu tls behaviour
    # it expensive (initialize on first use). gnu2 gives a significant performance boost.
    cc_args = ["-mavx", "-mavx2", "-O3", "-std=c++20", "-mfma", "-march=native", "-mtls-dialect=gnu2"]
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
    name="meaningful_spot_detector",
    version="1.0.0",
    url='https://github.com/axeldavy/MeaningfulSpot',
    license='MIT',
    python_requires='>=3.10',
    ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3", 'freethreading_compatible': True})
)