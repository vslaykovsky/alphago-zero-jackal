cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -g -O1")

project(jackal)


# torch
set(Torch_DIR /home/vslaykovsky/Downloads/libtorch-cxx11-abi-shared-with-deps-1.8.1+cu111/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# opencv
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# json
find_package(nlohmann_json REQUIRED)
include_directories()

file(GLOB_RECURSE SRCS src/tictactoe/*.cpp src/jackal/*.cpp src/mcts/*.cpp src/rl/*.cpp src/util/*.cpp third_party/tb_logger/src/*.cc)
link_libraries(stdc++fs ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Protobuf_LIBRARIES})
include_directories(third_party/tb_logger/include)

# perftest
#add_executable(perftest src/perftest.cpp)
#target_link_libraries(perftest "${TORCH_LIBRARIES}")
#set_property(TARGET perftest PROPERTY CXX_STANDARD 14)

#boost
#find_package(Boost REQUIRED)
#include_directories( ${Boost_INCLUDE_DIRS} )

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

# rl
#add_executable(self_play src/jackal_simple_self_play.cpp ${SRCS})
#target_link_libraries(self_play ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#set_property(TARGET self_play PROPERTY CXX_STANDARD 14)




#alphago_zero
add_executable(alphago_zero src/alphago_zero.cpp ${SRCS})
target_link_libraries(alphago_zero nlohmann_json::nlohmann_json  ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${Protobuf_LIBRARIES} pthread)
set_property(TARGET alphago_zero PROPERTY CXX_STANDARD 14)


# testing
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIR})

file(GLOB TEST_SRCS tests/*.cpp)
add_executable(unittests ${TEST_SRCS} ${SRCS})
target_link_libraries(unittests gtest_main gtest)
add_test(
        NAME unittests
        COMMAND unittests
        WORKING_DIRECTORY /home/vslaykovsky/CLionProjects/jackal_cpp
)

