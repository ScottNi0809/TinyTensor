cmake_minimum_required(VERSION 3.20)
project(TinyTensor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(tinytensor
    src/tensor.cpp
    src/aligned_alloc.cpp
    src/thread_pool.cpp
)
target_include_directories(tinytensor PUBLIC include)

add_executable(01_create_and_copy examples/01_create_and_copy.cpp)
target_link_libraries(01_create_and_copy PRIVATE tinytensor)

add_executable(02_aos_soa_transform examples/02_aos_soa_transform.cpp)
target_link_libraries(02_aos_soa_transform PRIVATE tinytensor)

add_executable(03_async_pipeline examples/03_async_pipeline.cpp)
target_link_libraries(03_async_pipeline PRIVATE tinytensor)

# 可执行输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
