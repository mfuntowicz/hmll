cmake_minimum_required(VERSION 3.16.0)

project(hmll LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

###########################################################
# Build Arguments
###########################################################
option(HMLL_ENABLE_CUDA "Flag to build with CUDA capabilities" OFF)
option(HMLL_ENABLE_PYTHON "Flag to build Python binding" OFF)
option(HMLL_BUILD_STATIC "Flag indicating if cmake should generate a static build for libhmll" OFF)
option(HMLL_BUILD_EXAMPLES "Flag indicating if cmake should build examples" OFF)
option(HMLL_BUILD_TESTS "Flag indicating if cmake should build unit-tests" OFF)
option(HMLL_ENABLE_SANITIZERS "Flag indicating if cmake should enable sanitizers" OFF)
option(HMLL_ENABLE_IO_URING "Flag to include io_uring backend (only on Linux)" ON)

if(${HMLL_BUILD_STATIC})
    message(STATUS "Building libhmll statically")
    set(HMLL_STATIC ON)
endif()

###########################################################
# Dependencies
###########################################################
set(HMLL_LIBRARIES "")
fetchcontent_declare(
        yyjson
        GIT_REPOSITORY https://github.com/ibireme/yyjson.git
        GIT_TAG        master
)
fetchcontent_makeavailable(yyjson)
list(APPEND HMLL_LIBRARIES yyjson)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${HMLL_ENABLE_IO_URING})
    include(cmake/liburing.cmake)
    list(APPEND HMLL_LIBRARIES liburing_static)
    add_compile_definitions("_GNU_SOURCE")
    message(STATUS "Enabling io_uring backend")
endif()


if(${HMLL_ENABLE_CUDA})
    find_package(CUDAToolkit REQUIRED)
    list(APPEND HMLL_LIBRARIES CUDA::cudart)
    message(STATUS "Enabling CUDA support")
endif()


if(${HMLL_ENABLE_PYTHON})
    add_subdirectory(lib/python)
endif()

###########################################################
# Definition of libhmll
###########################################################
set(LIBHMLL_HEADERS
        lib/memory.c
        include/hmll/fetcher.h
        include/hmll/hmll.h
        include/hmll/safetensors.h
        include/hmll/types.h
        include/hmll/unix/mmap.h
        include/hmll/unix/fetcher.h
        include/hmll/unix/iouring.h
)
set(LIBHMLL_SOURCES
        lib/context.c
        lib/fetcher.c
        lib/tensors.c
        lib/safetensors.c
        lib/sources.c
        lib/unix/mmap.c
        lib/unix/iouring.c
        lib/errors.c
)
add_library(libhmll ${LIBHMLL_HEADERS} ${LIBHMLL_SOURCES})
target_include_directories(libhmll PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(libhmll PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall -Werror -Wextra -Wpedantic>)
target_link_libraries(libhmll ${HMLL_LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(libhmll PUBLIC DEBUG=1)
endif()

###########################################################
# Examples
###########################################################
if(${HMLL_BUILD_EXAMPLES})
    add_executable(hmll_safetensors_ex examples/safetensors.c)
    target_link_libraries(hmll_safetensors_ex PRIVATE libhmll)
endif()

###########################################################
# Unit-tests
###########################################################
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

    set(HMLL_TESTS_SRC
            tests/tests_hmll_status.cpp
            tests/tests_hmll_dtype.cpp
            tests/tests_hmll_tensor_specs.cpp)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${HMLL_ENABLE_IO_URING})
        list(APPEND HMLL_TESTS_SRC tests/tests_hmll_io_uring.cpp)
    endif()

    add_executable(libhmll_tests ${HMLL_TESTS_SRC})
    target_link_libraries(libhmll_tests PRIVATE Catch2::Catch2WithMain libhmll)
    catch_discover_tests(libhmll_tests)
endif()