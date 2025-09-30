cmake_minimum_required(VERSION 2.8.3)
project(object_pose_estimator)

find_package(catkin REQUIRED COMPONENTS
    std_msgs
)

catkin_package(
    CATKIN_DEPENDS 
)

include_directories(
    # include
    ${Boost_INCLUDE_DIR}
    ${catkin_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")


