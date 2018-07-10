# SixTrackLib

## Building and Installation
### Prerequisites and Preparation
- CMake 3.8.x or higher (https://cmake.org/download/)
- A C99 compatible compiler (gcc 6.x or higher recommended)
- A c++11 compatible compiler (g++ 6.x or higher recommended)
- _optionally_ googletest library for running the unit-tests
- _optionally_ NVidia CUDA 7.5 or higher 
- _optionally_ OpenCL 1.2 or higher (with C++ host bindings)

The default configuration builds a CPU only library without any examples or tests. To configure the building process
- Create a copy of `Settings.cmake.default` and name it `Settings.cmake`
- Change the settings / options from `Off` to `On` to enable building specific modules or condigurations
- Save the file

### Building
Within the top-level directory of the repository, 
```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/opt/sixtracklib
```
If you have enabled the building of unit tests in the settings file, you may have to pass the location of the googletest library location to cmake. Assuming you installed googletest with the install prefix `/opt/googletest`, please set the `SIXTRACKL_GOOGLETEST_ROOT` variable in `Settings.cmake` to this value.

Assuming cmake finished successfully, build and (optionally) install the library by running
```
make
make install
```
### Unit-Tests
If you configured sixtracklib with unit-test support enabled in the previous steps, you can run the tests from the build directory by 
```
make testdata
make test
```
Alternatively, you can run the unit-tests individually from the `tests/sixtracklib/` sub-directory within build. Please ensure to always have a proper set of testdata ready before running any test. If in doubt, please run `make testdata`.

## Documentation

A draft of the physics manual is being edited in  https://www.overleaf.com/read/jsfjffbnvhvl
