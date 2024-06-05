cmake_minimum_required(VERSION 3.13)
project(cinnamon VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# -----------------------------------------------------------------------------
# Setup dependencies.
# -----------------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
find_package(onnxruntime)

# -----------------------------------------------------------------------------
# Build Cinnamon Runtime library.
# -----------------------------------------------------------------------------
set(LIB_NAME cinnamon)
# Setup headers and sources directory.
set(LIB_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/modules)
set(LIB_PRIV_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/headers)
# Scan all source files.
file(GLOB LIB_SRC_FILES 
  ${LIB_SRC_DIR}/model/*.cpp
)
# Build library from headers and source files.
add_library(${LIB_NAME} SHARED ${LIB_HEADERS} ${LIB_PRIV_HEADERS} ${LIB_SRC_FILES})
# Link dependencies.
target_link_libraries(${LIB_NAME} onnxruntime)
# When library is exposed, use #include <cinnamon/abc.h>
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${LIB_HEADERS}> $<INSTALL_INTERFACE:include>)
# Internally, use #include "abc.h"
target_include_directories(${LIB_NAME} PRIVATE $<BUILD_INTERFACE:${LIB_HEADERS}/${LIB_NAME}> $<INSTALL_INTERFACE:include/${LIB_NAME}>)
# Set library properties.
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# -----------------------------------------------------------------------------
# Buil executable and tests.
# -----------------------------------------------------------------------------
add_executable(hello tests/hello.cpp)
target_link_libraries(hello cinnamon)