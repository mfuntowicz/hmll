fetchcontent_declare(fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 12.1.0
)

fetchcontent_makeavailable(fmt)

# Suppress warnings from fmt (third-party dependency)
if(TARGET fmt)
    if(MSVC)
        target_compile_options(fmt PRIVATE /w)
    else()
        target_compile_options(fmt PRIVATE -w)
    endif()
endif()

message(STATUS "Enabling fmt")