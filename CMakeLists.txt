cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 14)
project(shape_based_matching)

# SET(CMAKE_BUILD_TYPE "Release")
SET(CMAKE_BUILD_TYPE "Debug")
SET(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
find_package(OpenCV 3.4.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(SOURCE
    line2Dup.cpp)

add_executable(${PROJECT_NAME}_executable ${SOURCE} test.cpp)
target_link_libraries(${PROJECT_NAME}_executable ${OpenCV_LIBS})

