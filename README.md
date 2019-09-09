# SixTrackLib

## Building and Installation
### Prerequisites and Preparation
- CMake 3.8.x or higher (https://cmake.org/download/)
- A C99 compatible compiler (gcc 6.x or higher recommended)
- A c++11 compatible compiler (g++ 6.x or higher recommended)
- _optionally_ googletest library for running the unit-tests
- _optionally_ NVidia CUDA 7.5 or higher 
- _optionally_ OpenCL 1.2 or higher (with C++ host bindings)
- _optionally_ Python 3.7 or higher with numpy, scipy, cobjects, pysixtrack, and sixtracktools modules (for Python bindings)

The default configuration builds a CPU only library without any examples or tests. To configure the building process
- Create a copy of `Settings.cmake.default` and name it `Settings.cmake`
- Change the settings / options from `Off` to `On` to enable building specific modules or condigurations
- Save the file
- Before moving on, take care that the correct Python interpreter is visible to cmake at this stage if you have enabled the bindings (i.e. activate the virtualenv, etc.)

There are a number of external dependencies (e.g. the googletest library, Proper OpenCL 2.x and 1.x C++ headers, etc.) which if not found already on your system will by default be downloaded and build as part of this build process. Please modify the corresponding entries in your copy of the Settings.cmake file to influence the discovery of already present dependencies and/or to facilitate stand-alone offline builds. 

### Building
Within the top-level directory of the repository, 
```
mkdir build
cmake .. 
```
By default the library is build in release mode and uses the system-default install prefix. Use 
``` 
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=~/opt/sixtracklib
```
in case you need to influence these choices. 

If you have enabled the building of unit tests in the settings file, the build configuration will attempt to find a recent version of the googletest library on your system and, failing that, will try to download the latest version from github and build a version you may have to pass the location of the googletest library location to cmake. Assuming you installed googletest with the install prefix `/opt/googletest`, please set the `SIXTRACKL_GOOGLETEST_ROOT` variable in `Settings.cmake` to this value.

Assuming cmake finished successfully, build and (optionally) install the library by running
```
make
make install
```
### Installing SixTrackLib Python bindings
If you had enabled the Python bindings during the configuration step and have successfully completed the creation of the library itself, please proceed as follows to install the Python bindings:
- install numpy, scipy, and matplotlib
- install cobjects (https://github.com/SixTrack/cobjects)
- install pysixtrack( https://github.com/SixTrack/pysixtrack)
- install sixtracktools (https://github.com/SixTrack/sixtracktools)
- move from the build directoy to the `python` subdirectory, re-run cmake to update the `python` installer templates with the most recent versions of the library from the last build and install using pip:
```
pwd # verify that we are in the build directory
cmake ..
cd ../python
pip install -e .
```
You can use the bindings from your Python script by means of 
```
python
Python 3.7.1 (default, Oct 22 2018, 11:21:55) 
[GCC 8.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sixtracklib
>>> 
```

### Unit-Tests
If you configured sixtracklib with unit-test support enabled in the previous steps, you can run the tests from the build directory by 
```
pwd # verify that we are in the build directory
make test
```
Alternatively, you can run the unit-tests individually from the `tests/sixtracklib/` and `tests/python` sub-directories within build. Please ensure to always have a proper set of testdata ready before running any test. If in doubt, please run `make testdata`.

## Documentation

A draft of the physics manual is being edited in  https://www.overleaf.com/read/jsfjffbnvhvl
