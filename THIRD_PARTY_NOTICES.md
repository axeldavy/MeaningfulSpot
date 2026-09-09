# Third-Party Notices

This project is distributed under the MIT License (see `LICENSE`) and includes source code from third-party projects under their own licenses.

## Bundled source code

1. Pylene (subset vendored under `externals/pylene`)
- License: Mozilla Public License 2.0 (MPL-2.0)
- Copyright: Contributors to Pylene
- License text: `externals/pylene/LICENSE`

2. xsimd headers (vendored under `externals/xsimd`)
- License: BSD 3-Clause License
- Copyright: xsimd contributors
- License text: `externals/xsimd/LICENSE`

## Build dependencies used in full build

1. oneTBB
- License: Apache-2.0

2. fmt
- License: MIT

3. boost, Eigen, range-v3, hwloc
- Used as dependencies during build.
- Their licenses are provided by their upstream projects and Conan recipes.

## Notes

- This repository excludes Pylene image I/O sources from the built extension.
- If you redistribute binaries, keep this notice and bundled license files with the distribution.
