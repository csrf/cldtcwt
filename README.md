# Dual-tree Complex Wavelet Transform in OpenCL

[![Build Status](https://travis-ci.org/csrf/cldtcwt.png?branch=master)](https://travis-ci.org/csrf/cldtcwt)

This repository contains code to perform the dual-tree complex wavelet
transform in an accelerated manner user OpenCL. It also includes an
implementation of polar-matching based keypoint descriptors.

## Building

### Quick start

Go from zero to deployed in three lines:

```console
$ git clone https://github.com/csrf/cldtcwt.git
$ cd cldtcwt; mkdir build; cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$PWD/../deploy .. && make all test install
```

This will install the cldtcwt library to the `deploy/` directory in the project
root. You will need to update your `PATH` and `LD_LIBRARY_PATH` environment
variables:

```console
$ # In cldtcwt project root directory
$ export PATH=$PWD/deploy/bin:$PATH
$ export LD_LIBRARY_PATH=$PWD/deploy/lib:$LD_LIBRARY_PATH
```

The test suite should've tested your build and your installation can be tested
by attempting to run the `displayVideoDTCWT` program passing it a
FFMPEG-playable video file:

```console
$ displayVideoDTCWT /path/to/video.mp4
```

### Dependencies

This library uses the [CMake](http://cmake.org) build system. Required
dependencies include:

* OpenCL implementation
* OpenGL implementation
* [SFML](http://www.sfml-dev.org/)
* [FFMPEG](http://ffmpeg.org/)
* [HDF5](http://www.hdfgroup.org/HDF5/) C library and C++ interface
* [Eigen](http://eigen.tuxfamily.org/) Linear Algebra library for C++

CMake will complain at configuration time if one or more of these dependencies
cannot be found.

## Licensing 

This project is licensed under the terms of the GNU General Public Licence v2;
see the COPYING file for further information. Information on the authorship of
the library can be found in the AUTHORS.md file.
