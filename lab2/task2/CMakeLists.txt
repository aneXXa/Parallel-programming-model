cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(lab2_task2 VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(integral_omp task2.cpp)

# OpenMP (required for parallel version)
find_package(OpenMP REQUIRED)
target_link_libraries(integral_omp PRIVATE OpenMP::OpenMP_CXX)

if(MSVC)
  target_compile_options(integral_omp PRIVATE /O2)
else()
  target_compile_options(integral_omp PRIVATE -O2)
endif()

