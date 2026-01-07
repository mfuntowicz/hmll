//
// Created by mfuntowicz on 12/1/25.
//

#include <catch2/catch_all.hpp>

#include "hmll/hmll.h"

TEST_CASE("HMLL_OK macro creates success error", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE(err.code == HMLL_ERR_SUCCESS);
    REQUIRE(err.sys_err == 0);
}

TEST_CASE("HMLL_ERR macro creates library error", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    REQUIRE(err.code == HMLL_ERR_ALLOCATION_FAILED);
    REQUIRE(err.sys_err == 0);
}

TEST_CASE("HMLL_SYS_ERR macro creates system error", "[error]")
{
    auto err = HMLL_SYS_ERR(42);
    REQUIRE(err.code == HMLL_ERR_SYSTEM);
    REQUIRE(err.sys_err == 42);
}

TEST_CASE("hmll_check detects success", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE_FALSE(hmll_check(err));
}

TEST_CASE("hmll_check detects library error", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    REQUIRE(hmll_check(err));
}

TEST_CASE("hmll_check detects system error", "[error]")
{
    auto err = HMLL_SYS_ERR(5);
    REQUIRE(hmll_check(err));
}

TEST_CASE("hmll_check detects error with both code and sys_err set", "[error]")
{
    auto err = HMLL_RES(.code = HMLL_ERR_IO_ERROR, .sys_err = 13);
    REQUIRE(hmll_check(err));
}

TEST_CASE("hmll_success validates success state", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE(hmll_success(err));
}

TEST_CASE("hmll_success rejects library error", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);
    REQUIRE_FALSE(hmll_success(err));
}

TEST_CASE("hmll_success rejects system error", "[error]")
{
    auto err = HMLL_SYS_ERR(2);
    REQUIRE_FALSE(hmll_success(err));
}

TEST_CASE("hmll_success rejects mixed error state", "[error]")
{
    auto err = HMLL_RES(.code = HMLL_ERR_MMAP_FAILED, .sys_err = 12);
    REQUIRE_FALSE(hmll_success(err));
}

TEST_CASE("hmll_error_is_os_error detects system errors", "[error]")
{
    auto err = HMLL_SYS_ERR(1);
    REQUIRE(hmll_error_is_os_error(err));
}

TEST_CASE("hmll_error_is_os_error rejects library errors", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_BUFFER_TOO_SMALL);
    REQUIRE_FALSE(hmll_error_is_os_error(err));
}

TEST_CASE("hmll_error_is_os_error rejects success", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE_FALSE(hmll_error_is_os_error(err));
}

TEST_CASE("hmll_error_is_lib_error detects library errors", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_TENSOR_NOT_FOUND);
    REQUIRE(hmll_error_is_lib_error(err));
}

TEST_CASE("hmll_error_is_lib_error rejects system errors", "[error]")
{
    auto err = HMLL_SYS_ERR(10);
    REQUIRE_FALSE(hmll_error_is_lib_error(err));
}

TEST_CASE("hmll_error_is_lib_error rejects success", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE_FALSE(hmll_error_is_lib_error(err));
}

TEST_CASE("hmll_strerr returns message for success", "[error]")
{
    auto err = HMLL_OK;
    const char* msg = hmll_strerr(err);
    REQUIRE(msg != nullptr);
}

TEST_CASE("hmll_strerr returns message for all library error codes", "[error]")
{
    const hmll_status_code codes[] = {
        HMLL_ERR_UNSUPPORTED_PLATFORM,
        HMLL_ERR_UNSUPPORTED_FILE_FORMAT,
        HMLL_ERR_UNSUPPORTED_DEVICE,
        HMLL_ERR_ALLOCATION_FAILED,
        HMLL_ERR_TABLE_EMPTY,
        HMLL_ERR_TENSOR_NOT_FOUND,
        HMLL_ERR_INVALID_RANGE,
        HMLL_ERR_BUFFER_ADDR_NOT_ALIGNED,
        HMLL_ERR_BUFFER_TOO_SMALL,
        HMLL_ERR_IO_ERROR,
        HMLL_ERR_FILE_NOT_FOUND,
        HMLL_ERR_FILE_EMPTY,
        HMLL_ERR_MMAP_FAILED,
        HMLL_ERR_IO_BUFFER_REGISTRATION_FAILED,
        HMLL_ERR_SAFETENSORS_JSON_INVALID_HEADER,
        HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER,
        HMLL_ERR_SAFETENSORS_JSON_MALFORMED_INDEX,
        HMLL_ERR_CUDA_NOT_ENABLED,
        HMLL_ERR_CUDA_NO_DEVICE,
        HMLL_ERR_UNKNOWN_DTYPE
    };

    for (auto code : codes) {
        auto err = HMLL_ERR(code);
        const char* msg = hmll_strerr(err);
        REQUIRE(msg != nullptr);
        REQUIRE(strlen(msg) > 0);
    }
}

TEST_CASE("hmll_strerr returns message for system error", "[error]")
{
    auto err = HMLL_SYS_ERR(2);
    const char* msg = hmll_strerr(err);
    REQUIRE(msg != nullptr);
    REQUIRE(strlen(msg) > 0);
}

TEST_CASE("hmll_error_is_lib_error detects library error even with sys_err set", "[error]")
{
    auto err = HMLL_RES(.code = HMLL_ERR_ALLOCATION_FAILED, .sys_err = 5);
    REQUIRE(hmll_error_is_lib_error(err));
}

TEST_CASE("hmll_error_is_os_error detects system error even with code set", "[error]")
{
    auto err = HMLL_RES(.code = HMLL_ERR_IO_ERROR, .sys_err = 13);
    REQUIRE(hmll_error_is_os_error(err));
}

TEST_CASE("hmll_strerr returns specific message for mapped error codes", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);
    const char* msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "File not found") == 0);

    err = HMLL_ERR(HMLL_ERR_ALLOCATION_FAILED);
    msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "Failed to allocate memory") == 0);

    err = HMLL_ERR(HMLL_ERR_TABLE_EMPTY);
    msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "No tensors found while reading the file") == 0);

    err = HMLL_ERR(HMLL_ERR_TENSOR_NOT_FOUND);
    msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "Tensor not found in the known tensors table") == 0);

    err = HMLL_ERR(HMLL_ERR_CUDA_NOT_ENABLED);
    msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "CUDA not enabled") == 0);

    err = HMLL_ERR(HMLL_ERR_CUDA_NO_DEVICE);
    msg = hmll_strerr(err);
    REQUIRE(strcmp(msg, "No CUDA devices found") == 0);
}

TEST_CASE("hmll_strerr returns unknown error message for unmapped error codes", "[error]")
{
    auto err = HMLL_ERR(HMLL_ERR_BUFFER_TOO_SMALL);
    const char* msg = hmll_strerr(err);
    REQUIRE(strstr(msg, "Unknown error") != nullptr);
}

TEST_CASE("hmll_strerr handles negative sys_err values", "[error]")
{
    auto err = HMLL_SYS_ERR(-2);
    const char* msg = hmll_strerr(err);
    REQUIRE(msg != nullptr);
    REQUIRE(strlen(msg) > 0);
}

TEST_CASE("hmll_check and hmll_success are logical inverses", "[error]")
{
    auto err = HMLL_OK;
    REQUIRE(hmll_check(err) == !hmll_success(err));

    err = HMLL_ERR(HMLL_ERR_FILE_NOT_FOUND);
    REQUIRE(hmll_check(err) == !hmll_success(err));

    err = HMLL_SYS_ERR(5);
    REQUIRE(hmll_check(err) == !hmll_success(err));

    err = HMLL_RES(.code = HMLL_ERR_MMAP_FAILED, .sys_err = 12);
    REQUIRE(hmll_check(err) == !hmll_success(err));
}