cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(lab2 VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(dgemv_omp task1.cpp)

# OpenMP (required for parallel version)
find_package(OpenMP REQUIRED)
target_link_libraries(dgemv_omp PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(dgemv_omp PRIVATE /O2)
else()
  target_compile_options(dgemv_omp PRIVATE -O2)
endif()
