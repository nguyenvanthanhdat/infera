cmake_minimum_required(VERSION 3.13)
project(cinnamon_runtime VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Include all cmake files in /cmake directory.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
# Find OnnxRuntime lib.
find_package(onnxruntime)
# Make executable
add_executable(hello hello.cpp)
# Link libraries.
target_include_directories(hello PUBLIC include)
target_link_directories(hello PRIVATE src)
target_link_libraries(hello ${onnxruntime_LIBRARY})