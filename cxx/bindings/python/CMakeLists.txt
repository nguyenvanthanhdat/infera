cmake_minimum_required(VERSION 3.13)
project(cinnamon VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)

# Include all cmake files in /cmake directory.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/onnxruntimeConfig.cmake)
# Find dependencies.
find_package(onnxruntime REQUIRED)
find_package(pybind11 REQUIRED)

# Make library
INCLUDE(GNUInstallDirs)
set(PUBLIC_HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
#
set(PRIVATE_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../core)
set(LIBRARY_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../core)
#
add_library(${PROJECT_NAME} SHARED 
  ${PRIVATE_HEADER_FILES} ${LIBRARY_SOURCE_FILES}/model.cpp
  ${PRIVATE_HEADER_FILES} ${LIBRARY_SOURCE_FILES}/modelManager.cpp
  ${PRIVATE_HEADER_FILES} ${LIBRARY_SOURCE_FILES}/serviceManager.cpp)
# add_library(${PROJECT_NAME} SHARED ${PRIVTAE_HEADER_FILES} ${PUBLIC_HEADER_FILES})
#
target_include_directories(${PROJECT_NAME} PRIVATE 
  $<BUILD_INTERFACE:${PRIVATE_HEADER_FILES}> 
  $<INSTALL_INTERFACE:include/${PROJECT_NAME}>
)
#
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${PUBLIC_HEADER_FILES}>
  $<INSTALL_INTERFACE:include>
)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Make executable
# add_executable(hello_cinnamon tests/hello.cpp)
# add_dependencies(hello_cinnamon ${PROJECT_NAME})
pybind11_add_module(ortcxx MODULE main.cpp)
add_dependencies(ortcxx ${PROJECT_NAME})
# Link libraries.
# target_link_libraries(ortcxx ${onnxruntime_LIBRARY} ${PROJECT_NAME} pybind_lib)
target_link_libraries(ortcxx PRIVATE ${PROJECT_NAME} ${onnxruntime_LIBRARY} pybind11::module)