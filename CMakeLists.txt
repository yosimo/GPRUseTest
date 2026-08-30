cmake_minimum_required(VERSION 3.14)

project(gpr_use_test LANGUAGES CXX)

include(CTest)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Override from the command line with -DLBFGSPP_ROOT=/path/to/LBFGSpp.
set(LBFGSPP_ROOT "$ENV{HOME}/lib/LBFGSpp" CACHE PATH
    "Path to the LBFGSpp source directory")
find_path(LBFGSPP_INCLUDE_DIR
    NAMES LBFGSB.h
    HINTS "${LBFGSPP_ROOT}/include"
    REQUIRED
)

add_library(surrogate STATIC
    surrogate/src/config.cpp
    surrogate/src/gaussian_process.cpp
    surrogate/src/hybrid_regression.cpp
    surrogate/src/input_transform.cpp
    surrogate/src/polynomial_trend.cpp
    surrogate/src/regression_dataset.cpp
)
target_compile_features(surrogate PUBLIC cxx_std_17)
target_include_directories(surrogate
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/surrogate/include"
    PRIVATE
        "${LBFGSPP_INCLUDE_DIR}"
)
target_link_libraries(surrogate PUBLIC Eigen3::Eigen)

add_executable(gpr_test main.cpp)
target_link_libraries(gpr_test PRIVATE surrogate)

add_executable(
    hybrid_regression_example
    examples/hybrid_regression_example.cpp
)
target_link_libraries(hybrid_regression_example PRIVATE surrogate)

if(BUILD_TESTING)
    add_executable(
        hybrid_regression_tests
        tests/hybrid_regression_tests.cpp
    )
    target_link_libraries(hybrid_regression_tests PRIVATE surrogate)
    add_test(
        NAME hybrid_regression_tests
        COMMAND hybrid_regression_tests
    )
endif()
