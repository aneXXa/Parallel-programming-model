cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(lab3_task1_stdthread VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(task1_stdthread task1.cpp)

if(MSVC)
  target_compile_options(task1_stdthread PRIVATE /O2)
else()
  target_compile_options(task1_stdthread PRIVATE -O2 -pthread)
endif()

enable_testing()

add_test(
  NAME task1_full_check
  COMMAND task1_stdthread
    --sizes 500,800
    --threads 1,2,4,7,8,16,20,40
    --repeats 2
    --drop-max 0
    --out-prefix ctest_speedup_dgemv_stdthread
)

add_custom_target(
  run_full_benchmark
  COMMAND task1_stdthread
    --sizes 20000,40000
    --threads 1,2,4,7,8,16,20,40
    --repeats 5
    --drop-max 1
  DEPENDS task1_stdthread
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Run full benchmark without Python scripts"
)
