This repository contains the reference implementation for my paper MEANINGFUL LEVEL SETS FOR SMALL SPOT DETECTION, accepted at the 2026 IEEE International Conference on Image Processing (ICIP).

# Warning

This repository is still a WIP and some changes may be made (method signature, etc) before
the paper is presented at ICIP.

# Building

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

In order to test with Python, setup.py builds a Python module that provides various methods to call
both implementation (see spot_detector_cpp.pyi for their signatures). An alternative setup that
doesn't compile the Max-Tree implementation (setup_no_maxtree.py) is provided and enables to test
the algorithm without needing all the dependencies of Pylene. For that alternative setup, all calls
to the Max-Tree implementation are diverted to the local level set search variant.

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