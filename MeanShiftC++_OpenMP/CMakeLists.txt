cmake_minimum_required(VERSION 3.25)
project(untitled)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(untitled main.cpp Cluster.cpp Cluster.h ClusterManager.cpp ClusterManager.h MeanShift.cpp MeanShift.h Point.cpp Point.h SpeedUpTests.cpp SpeedUpTests.h Utils.cpp Utils.h)

target_link_libraries(untitled -lstdc++fs)