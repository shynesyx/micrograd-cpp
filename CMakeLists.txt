cmake_minimum_required(VERSION 3.10)
project(micrograd_cpp)

set(EXE ${PROJECT_NAME})
set(SOURCE_DIR micrograd)
set(TESTS_DIR tests)

set(CMAKE_BUILD_TYPE Debug)

# include directories
include_directories(${SOURCE_DIR}) 

# Find all .cpp files in src directory
file(GLOB SOURCE_FILES ${SOURCE_DIR}/*.cpp)
add_executable(${EXE} ${SOURCE_FILES} main.cpp)

# Add a target to clean the build folder
add_custom_target(clean_build
  COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles
  COMMAND echo "Cleaning complete!"
)


# ----------
# Test Stuff
# ----------

# GoogleTest related
# requires at least C++14 (starting from v1.13.0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# dependency
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/b514bdc898e2951020cbdca1304b75f5950d1f59.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(TEST_EXE micrograd_cpp_test)

# Find all .cpp files in tests directory
file(GLOB TEST_SRCS ${TESTS_DIR}/*.cpp)

add_executable(${TEST_EXE} ${SOURCE_FILES} ${TEST_SRCS})
target_link_libraries(${TEST_EXE} PUBLIC GTest::gtest GTest::gmock_main GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${TEST_EXE})
