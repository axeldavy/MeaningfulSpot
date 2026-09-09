This repository contains the reference implementation for my paper MEANINGFUL LEVEL SETS FOR SMALL SPOT DETECTION, accepted at the 2026 IEEE International Conference on Image Processing (ICIP).

# License

This project is distributed under the MIT License (see `LICENSE`).
It also bundles third-party code with its own licenses; see `THIRD_PARTY_NOTICES.md`.

# Warning

This repository is still a WIP and some changes may be made (method signature, etc) before
the paper is presented at ICIP.

# Building and installing locally

The project exposes a single normal entry point for local development: the repository-root `setup.py`.
The actual compiler logic lives in `src/setup_no_maxtree.py` and `src/setup.py`; the root script is only a dispatcher.

The code is implemented in C++. As described in the paper, two variants are implemented:

These files contain the local level set search variant:
* bitarray.hpp
* maxpool.cpp
* maxpool.hpp
* meaningful_ll.cpp
* meaningful_ll.hpp
* spot_detector.cpp
* spot_detector.hpp

These files contain the implementation based on a Max-Tree:
* maxpool.cpp
* maxpool.hpp
* spot_detector_maxtree.cpp
* spot_detector_maxtree.hpp

The latter requires [Pylene](https://github.com/GerHobbelt/pylene) (which is included in externals) and is harder to build (Pylene adds dependencies).
Both require builds [xsimd](https://github.com/xtensor-stack/xsimd), which is a header-only library and which is included in externals as well.

## Local installation

For most users, the default minimal install is the easiest path:

```bash
python -m pip install -e .
```

This uses the repository-root `setup.py`, which defaults to the minimal build (`MEANINGFUL_SPOT_BUILD=minimal`). It compiles only the local level-set implementation and avoids the heavier Max-Tree dependency chain.

If you want the full Max-Tree implementation, set the build selector explicitly:

```bash
MEANINGFUL_SPOT_BUILD=full python -m pip install -e .
```

This keeps the native C++ dependencies managed automatically by Conan during the build, so users are not expected to install system libraries manually.

The full build intentionally excludes Pylene image file I/O sources and the related codec stack, because the detector does not use image loading at runtime.

For a direct in-place build without installing as an editable package:

```bash
python setup.py build_ext --inplace
```

or, for the full build:

```bash
MEANINGFUL_SPOT_BUILD=full python setup.py build_ext --inplace
```

The `pyproject.toml` file contains the project metadata and build-system requirements. The `src/setup_no_maxtree.py` and `src/setup.py` files are the actual compile scripts. In normal use, users should rely on the repository-root `setup.py` and the `MEANINGFUL_SPOT_BUILD` environment variable rather than invoking the `src` scripts by hand.

# Reproducing the paper

The publication includes comparisons on various datasets against other methods. The related code will be contained in a separate repository at a later date.

# Testing

Both the Max-Tree and the local level set search implementations provide the same results. However
the Max-Tree implementation should be faster when allowing large spots (large `n_elems`), or when
having a scene with very dense detections. The local level set search variant should be faster when
few detections are expected and `n_elems` is small.

See spot_detector_cpp.pyi for the documentation of available methods. 

For instance, one use would be:

```python
my_detected_boxes = detect_spots_as_boxes(my_image, noise_level=sigma)
```