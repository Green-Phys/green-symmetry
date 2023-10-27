project(symmetry_test)

Include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG v3.4.0 # or a later release
)

FetchContent_MakeAvailable(Catch2)
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

add_executable(symmetry_test symmetry_test.cpp)
target_compile_definitions(symmetry_test PRIVATE TEST_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(symmetry_test
        PRIVATE
        Catch2::Catch2WithMain
        GREEN::SYMMETRY)

include(CTest)
include(Catch)
catch_discover_tests(symmetry_test)
