find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
fetchcontent_declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG v2.10.2
)
fetchcontent_makeavailable(nanobind)

nanobind_add_module(_pyhmll_impl context.cpp specs.cpp safetensors.cpp pyhmll.cpp)
target_link_libraries(_pyhmll_impl PRIVATE libhmll)
set_target_properties(_pyhmll_impl PROPERTIES CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_pyhmll_impl PRIVATE -g -O0)
endif()
message(STATUS "Enabling Python binding")