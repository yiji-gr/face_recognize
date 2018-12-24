cmake_minimum_required(VERSION 2.8)

project(mtcnn_ncnn C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(/home/yiji/gr/ncnn/build/install/include /usr/include/eigen3)

link_directories(/home/yiji/gr/ncnn/build/install/lib)

file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

add_executable(mtcnn_ncnn ${MTCNN_COMPILE_CODE})

target_link_libraries(mtcnn_ncnn /home/yiji/gr/ncnn/build/install/lib/libncnn.a ${OpenCV_LIBS})
