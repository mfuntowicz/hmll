find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

### Nanobind to expose C++/C library to Python
fetchcontent_declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind
    GIT_TAG v2.10.2
)
fetchcontent_makeavailable(nanobind)

nanobind_add_module(_pyhmll_impl NB_STATIC STABLE_ABI LTO context.cpp fetcher.cpp specs.cpp safetensors.cpp pyhmll.cpp)
nanobind_add_stub(
        _pyhmll_impl_stub
        MODULE _pyhmll_impl
        OUTPUT _pyhmll_impl.pyi
        PYTHON_PATH $<TARGET_FILE_DIR:_pyhmll_impl>
        DEPENDS _pyhmll_impl
)
target_link_libraries(_pyhmll_impl PRIVATE libhmll)
set_target_properties(_pyhmll_impl PROPERTIES CXX_STANDARD 20)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_pyhmll_impl PRIVATE -g -O0)
else()
    target_compile_options(_pyhmll_impl PRIVATE -g -O3)
endif()
message(STATUS "Enabling Python binding")