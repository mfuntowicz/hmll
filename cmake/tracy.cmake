fetchcontent_declare(
    tracy
    GIT_REPOSITORY https://github.com/wolfpld/tracy.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
fetchcontent_makeavailable(tracy)
add_compile_definitions(__HMLL_PROFILE_ENABLED__=1)