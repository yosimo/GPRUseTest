cmake_minimum_required(VERSION 3.14)

project(gpr_use_test LANGUAGES CXX)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# コマンドラインから -DLBFGSPP_ROOT=/path/to/LBFGSpp で変更できる。
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
