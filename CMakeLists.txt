cmake_minimum_required(VERSION 3.10)

project(retina_face)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

# opencv
set( OpenCV_DIR /home/lhao/lib/clib/opencv/build)
find_package( OpenCV REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})

# cuda
include_directories("/usr/local/cuda/include")
link_directories("/usr/local/cuda/lib64")


# tensorrt
include_directories("/home/lhao/lib/clib/TensorRT-8.0.3.4/include" )
link_directories("/home/lhao/lib/clib/TensorRT-8.0.3.4/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/retina_face.cpp)
target_link_libraries(${PROJECT_NAME} nvinfer)
target_link_libraries(${PROJECT_NAME} cudart)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})


add_definitions(-O0 -pthread)