cmake_minimum_required(VERSION 3.50)

set(CXX_DISABLE_WERROR 1)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_NAME MinimumJerkTask)
set(PROJECT_DESCRIPTION "MinimJerk task for mc_rtc")
set(PROJECT_URL "")

set(PROJECT_USE_CMAKE_EXPORT TRUE)

include(cmake/base.cmake)

project(${PROJECT_NAME} LANGUAGES CXX VERSION 0.1.0)

if(NOT TARGET mc_rtc::control)
  find_package(mc_rtc REQUIRED)
endif()

enable_testing()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

include_directories(include)

add_subdirectory(src)
