cmake_minimum_required(VERSION 3.16.0)

project(hmll LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g ")
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(FetchContent)

###########################################################
# Build Arguments
###########################################################
option(HMLL_ENABLE_CUDA "Flag to build with CUDA capabilities" OFF)
option(HMLL_ENABLE_PYTHON "Flag to build Python binding" OFF)
option(HMLL_ENABLE_SAFETENSORS "Flag to add support for reading safetensors natively" OFF)
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
if(${HMLL_ENABLE_CUDA})
    find_package(CUDAToolkit REQUIRED)
    list(APPEND HMLL_LIBRARIES CUDA::cudart CUDA::cuda_driver)
    add_compile_definitions(libhmll PUBLIC __HMLL_CUDA_ENABLED__=1)
    message(STATUS "Enabling CUDA support")
endif()


if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${HMLL_ENABLE_IO_URING})
    include(cmake/liburing.cmake)
    list(APPEND HMLL_LIBRARIES liburing_static)
    add_compile_definitions("_GNU_SOURCE")
    message(STATUS "Enabling io_uring backend")
endif()


if(${HMLL_ENABLE_PYTHON})
    add_subdirectory(lib/python)
endif()

###########################################################
# Definition of libhmll
###########################################################
set(LIBHMLL_HEADERS
        include/hmll/hmll.h
        include/hmll/loader.h
        include/hmll/memory.h
        include/hmll/types.h
        include/hmll/unix/loader.h
        include/hmll/unix/backend/iouring.h
)
set(LIBHMLL_SOURCES
        lib/context.c
        lib/errors.c
        lib/loader.c
        lib/memory.c
#        lib/tensors.c
        lib/unix/loader.c
        lib/unix/file.c
        lib/unix/backend/iouring.c
)

if(${HMLL_ENABLE_SAFETENSORS})
    message(STATUS "Enabling safetensors format support")
    add_compile_definitions(libhmll PUBLIC __HMLL_SAFETENSORS_ENABLED__=1)
    add_compile_definitions(libhmll PUBLIC __HMLL_TENSORS_ENABLED__=1)
    fetchcontent_declare(
            yyjson
            GIT_REPOSITORY https://github.com/ibireme/yyjson.git
            GIT_TAG        master
    )
    fetchcontent_makeavailable(yyjson)

    list(APPEND LIBHMLL_SOURCES lib/safetensors.c lib/tensors.c)
    list(APPEND HMLL_LIBRARIES yyjson)
endif()

if(${HMLL_ENABLE_CUDA})
    list(APPEND LIBHMLL_HEADERS include/hmll/cuda.h)
endif()

add_library(libhmll ${LIBHMLL_HEADERS} ${LIBHMLL_SOURCES})
target_include_directories(libhmll PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_compile_options(libhmll PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall -Werror -Wextra -Wpedantic -Wno-unused-function>)
target_link_libraries(libhmll ${HMLL_LIBRARIES})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(libhmll PUBLIC DEBUG=1)
endif()

###########################################################
# Examples
###########################################################
if(${HMLL_BUILD_EXAMPLES})
    set(HMLL_EXAMPLES_EXTRA_LIBS "")
    if(${HMLL_ENABLE_CUDA})
        list(APPEND HMLL_EXAMPLES_EXTRA_LIBS CUDA::cudart)
    endif()

    if(${HMLL_ENABLE_SAFETENSORS})
        add_executable(hmll_safetensors_ex examples/safetensors.c)
        target_link_libraries(hmll_safetensors_ex PRIVATE libhmll ${HMLL_EXAMPLES_EXTRA_LIBS})
    endif()

    add_executable(hmll_examples_universal examples/universal.c)
    target_link_libraries(hmll_examples_universal PRIVATE libhmll ${HMLL_EXAMPLES_EXTRA_LIBS})
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
            GIT_TAG        v3.12.0
    )
    fetchcontent_makeavailable(Catch2)
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

    include(CTest)
    include(Catch)

    set(HMLL_TESTS_SRC
            tests/tests_hmll_errors.cpp
            tests/tests_hmll_memory.cpp
            tests/tests_hmll_context.cpp)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${HMLL_ENABLE_IO_URING})
        list(APPEND HMLL_TESTS_SRC tests/tests_hmll_io_uring.cpp)
    endif()

    if(${HMLL_ENABLE_SAFETENSORS})
        list(APPEND HMLL_TESTS_SRC
            tests/tests_hmll_tensor_specs.cpp
            tests/tests_hmll_dtype.cpp
            tests/tests_hmll_safetensors.cpp
        )
    endif()

    add_executable(libhmll_tests ${HMLL_TESTS_SRC})
    target_include_directories(libhmll_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)
    target_link_libraries(libhmll_tests PRIVATE Catch2::Catch2WithMain libhmll)
    catch_discover_tests(libhmll_tests)
endif()