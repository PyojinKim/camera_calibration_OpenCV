# project
cmake_minimum_required(VERSION 2.6)
project(monocular CXX C)


# package
find_package(OpenCV REQUIRED)


# header file
#set(H ${H}	include/core/fileIO.h)


# source file
set(SRC ${SRC} 	src/calibration.cpp)


# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")


# directories
set(includeDirs ${includeDirs} include)


# include directory
include_directories(${includeDirs} ${OpenCV_INCLUDE_DIRS})


# make release version
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg  -m64 -Ofast -flto -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Ofast -flto -march=native")


# demo program
add_executable(monocular ${SRC} ${H})
target_link_libraries (monocular png ${OpenCV_LIBS})
