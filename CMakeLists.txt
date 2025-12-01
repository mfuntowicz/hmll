cmake_minimum_required(VERSION 3.20.0)

project(hmll LANGUAGES C CXX)
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)

include(FetchContent)

# Build Arguments
option(HMLL_BUILD_STATIC "Flag indicating if cmake should generate a static build for libhmll" OFF)
option(HMLL_BUILD_EXAMPLES "Flag indicating if cmake should build examples" OFF)
option(HMLL_BUILD_TESTS "Flag indicating if cmake should build unit-tests" OFF)
option(HMLL_ENABLE_SANITIZERS "Flag indicating if cmake should enable sanitizers" OFF)

if(${HMLL_BUILD_STATIC})
    set(HMLL_STATIC ON)
endif()

# Dependencies
fetchcontent_declare(
        yyjson
        GIT_REPOSITORY https://github.com/ibireme/yyjson.git
        GIT_TAG        master
)
fetchcontent_makeavailable(yyjson)

# Definition of libhmll
set(LIBHMLL_HEADERS
        include/hmll/hmll.h
        include/hmll/safetensors.h
        include/hmll/status.h
)
set(LIBHMLL_SOURCES
    lib/hmll_safetensors.c
    lib/hmll_status.c
)
add_library(libhmll ${LIBHMLL_HEADERS} ${LIBHMLL_SOURCES})
target_include_directories(libhmll PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(libhmll PRIVATE $<$<COMPILE_LANGUAGE:C>:-Werror -Wpedantic>)

# Unit-tests
if(${HMLL_BUILD_TESTS})
    if(NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        message(FATAL_ERROR "Test should be run with Debug configuration, got: ${CMAKE_BUILD_TYPE}")
    endif()

    fetchcontent_declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG        v3.11.0
    )
    fetchcontent_makeavailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

    include(CTest)
    include(Catch)

    add_executable(libhmll_tests tests/tests_hmll_safetensors.cpp
            tests/test_hmll_status.cpp)
    target_link_libraries(libhmll_tests PRIVATE Catch2::Catch2WithMain libhmll)
    catch_discover_tests(libhmll_tests)
endif()