# compiler and options
CXX = g++
OPTS = -std=c++14 -O3 -Wall -Wextra -pedantic

# location of the Python header files

PYTHON_VERSION = 2.7
PYTHON_CPPFLAGS = -I/usr/include/python$(PYTHON_VERSION)
PYTHON_LDFLAGS = -L/usr/lib/python$(PYTHON_VERSION)/config

# location of the Boost include files and libraries

BOOST_CPPFLAGS = -I/usr/include
BOOST_LDFLAGS  = -L/usr/lib

# location of CUDA

CUDA_PATH=/opt/cuda
CUDA_CPPFLAGS = -I$(CUDA_PATH)/include
CUDA_LDFLAGS = -L$(CUDA_PATH)/lib64


# compile mesh classes
tracker.so: interface.o
	echo "Linking tracker..."
	$(CXX) $(OPTS) -shared -Wl,--export-dynamic interface.o $(BOOST_LDFLAGS) -lboost_python $(PYTHON_LDFLAGS) -lpython$(PYTHON_VERSION) $(CUDA_LDFLAGS) -lnvrtc -lcuda -lcudart -Wl,-rpath,$(CUDA_LDFLAGS) -o tracker.so

interface.o: interface.cpp $(wildcard *.h)
	echo "Compiling tracker..."
	gcc -E -P -D_GPUCODE='' -DCLGLOBAL='' -D_CUDA_HOST_DEVICE_=__device__ ../common/track.h | xxd -i > track.xxd
	$(CXX) $(OPTS) $(PYTHON_CPPFLAGS) $(BOOST_CPPFLAGS) $(CUDA_CPPFLAGS) -I../common -fPIC -c interface.cpp
	rm -f track.xxd

clean:
	rm -f interface.o tracker.so
