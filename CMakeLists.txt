cmake_minimum_required(VERSION 3.14)

project(gpr_use_test LANGUAGES CXX)

include(CTest)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Threads REQUIRED)

# The target environment is offline. Third-party headers are expected at
# fixed locations below $HOME/lib; CMake does not search or download them.
set(LOCAL_LIBRARY_ROOT "$ENV{HOME}/lib")
set(LBFGSPP_INCLUDE_DIR
    "${LOCAL_LIBRARY_ROOT}/LBFGSpp/include")
set(FAST_CPP_CSV_PARSER_INCLUDE_DIR
    "${LOCAL_LIBRARY_ROOT}/fast-cpp-csv-parser")

if(NOT EXISTS "${LBFGSPP_INCLUDE_DIR}/LBFGSB.h")
    message(FATAL_ERROR
        "Local LBFGS++ header was not found: "
        "${LBFGSPP_INCLUDE_DIR}/LBFGSB.h")
endif()
if(NOT EXISTS "${FAST_CPP_CSV_PARSER_INCLUDE_DIR}/csv.h")
    message(FATAL_ERROR
        "Local fast-cpp-csv-parser header was not found: "
        "${FAST_CPP_CSV_PARSER_INCLUDE_DIR}/csv.h")
endif()

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
target_include_directories(gpr_test
    PRIVATE
        "${FAST_CPP_CSV_PARSER_INCLUDE_DIR}"
)
target_link_libraries(
    gpr_test
    PRIVATE
        surrogate
        Threads::Threads
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/data")
configure_file(
    data/hybrid_training_4d.csv
    data/hybrid_training_4d.csv
    COPYONLY
)
configure_file(
    data/hybrid_test_4d.csv
    data/hybrid_test_4d.csv
    COPYONLY
)

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
