cmake_minimum_required(VERSION 2.8)

project(face_recognize C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(/home/yiji/gr/ncnn/build/install/include) #ncnn 包含目录路径

link_directories(/home/yiji/gr/ncnn/build/install/lib)        #ncnn 库目录路径

file(GLOB MTCNN_SRC ${CMAKE_CURRENT_LIST_DIR}/src/*.h ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)

set(MTCNN_COMPILE_CODE ${MTCNN_SRC})

add_executable(face_recognize ${MTCNN_COMPILE_CODE})

target_link_libraries(face_recognize /home/yiji/gr/ncnn/build/install/lib/libncnn.a ${OpenCV_LIBS}) #libncnn.a 路径
