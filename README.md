# SixTrackLib

## Building and Installation
### Prerequisites and Preparation
- CMake 3.8.x or higher (https://cmake.org/download/)
- A C99 compatible compiler (gcc 6.x or higher recommended)
- A c++11 compatible compiler (g++ 6.x or higher recommended)
- _optionally_ googletest library for running the unit-tests
- _optionally_ NVidia CUDA 7.5 or higher 
- _optionally_ OpenCL 1.2 or higher (with C++ host bindings)

The default configuration builds a CPU only library without any examples and tests. To condigure the building process
- Create a copy of Settings.cmake.default and name it Settings.cmake
- Change the settings / options from `Off` to `On` to enable building specific modules or condigurations
- Save the file

### Building

```
mkdir build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/opt/sixtracklib
```
If you have enabled the building of unit tests in the settings file, you may have to pass the location of the googletest library location to cmake. Please assuming you installed googletest with the install prefix `/opt/googletest`, please call cmake like this
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/opt/sixtracklib -DGTEST_ROOT=/opt/googletest
```
Alternatively, you can also set the GTEST_ROOT option in the settings file.

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
Alternatively, you can run the unit-tests individually from the `tests/sixtracklib/` sub-directory within build.


## Documentation

A draft of the physics manual is being edited in  https://www.overleaf.com/read/jsfjffbnvhvl
