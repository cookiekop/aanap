cmake_minimum_required(VERSION 3.9)
project(aanap)

set(CMAKE_CXX_STANDARD 11)

add_executable(aanap main.cpp aanap.cpp aanap.h blender.cpp blender.h homography_warp.cpp homography_warp.h compute_weight.cpp compute_weight.h samplePoint.cpp samplePoint.h)

find_package( OpenCV REQUIRED )

target_link_libraries(aanap ${OpenCV_LIBS})