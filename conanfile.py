from conan import ConanFile


class MeaningfulSpotConan(ConanFile):
    name = "meaningful-spot-detector"
    package_type = "static-library"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    generators = "CMakeToolchain", "CMakeDeps", "VirtualRunEnv"
    requires = [
        "boost/1.85.0",
        "eigen/3.4.0",
        "fmt/11.0.2",
        "hwloc/2.12.2",
        "onetbb/2021.7.0",
        "range-v3/0.12.0",
    ]

    def configure(self):
        self.options["hwloc"].shared = True
