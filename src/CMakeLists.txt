#
# Library for self-consistency procedure
#

project(symmetry-lib)

add_library(symmetry
        symmetry.cpp
)

find_package(Eigen3 3.3 NO_MODULE REQUIRED)
find_package(BLAS)
if (BLAS_FOUND)
    if (BLAS_LIBRARIES MATCHES ".*Accelerate")
        message(INFO " Using Apple BLAS")
    endif ()
    if (BLAS_LIBRARIES MATCHES ".*openblas")
        message(INFO " Using OpenBLAS")
        add_compile_definitions(USE_GEMM3M)
    endif ()
    add_compile_definitions(EIGEN_USE_BLAS)
endif ()



target_link_libraries(symmetry PUBLIC GREEN::H5PP GREEN::NDARRAY GREEN::PARAMS Eigen3::Eigen)
if(BLAS_FOUND)
    target_link_libraries(symmetry PUBLIC BLAS::BLAS)
endif(BLAS_FOUND)
target_include_directories(symmetry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
