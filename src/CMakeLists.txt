#
# Library for self-consistency procedure
#

project(symmetry-lib)

add_library(symmetry
        symmetry.cpp
)

find_package(Eigen3 REQUIRED)

target_link_libraries(symmetry PUBLIC GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS Eigen3::Eigen)
target_include_directories(symmetry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})