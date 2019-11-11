# OpenCL-CLHPP OpenCL API C++ Bindings, Embedded Into SixTrackLib
This directory contains an embedded version of the header-only
[https://github.com/KhronosGroup/OpenCL-CLHPP](OpenCL(TM) API C++ bindings) by KhronosGroup
for use within SixTrackLib. Copyright (c) 2008-2015 The Khronos Group Inc.
Please refer to the LICENSE.txt file and the OpenCL-CLHPP_README.md files for further information about
the headers themselves.

**Note**: Any system-level present OpenCL C++ API headers have preference over this
embedded version. Plus, we attempt to update the contents of this directory from upstream
every time cmake is run. This can be switched off by forcing offline builds
via the settings file. Please refer to `cmake/SetupOpenCL.cmake` for details.
