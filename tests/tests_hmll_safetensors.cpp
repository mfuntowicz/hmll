//
// Created by mfuntowicz on 1/8/26.
//

#include "catch2/catch_all.hpp"
#include "hmll/hmll.h"
#include "safetensors_internal.h"

TEST_CASE("safetensors dtype parsing", "[safetensors]")
{
    SECTION("boolean type") {
        REQUIRE(hmll_safetensors_dtype_from_str("BOOL", 4) == HMLL_DTYPE_BOOL);
    }

    SECTION("floating point types") {
        REQUIRE(hmll_safetensors_dtype_from_str("BF16", 4) == HMLL_DTYPE_BFLOAT16);
        REQUIRE(hmll_safetensors_dtype_from_str("F16", 4) == HMLL_DTYPE_FLOAT16);
        REQUIRE(hmll_safetensors_dtype_from_str("F32", 4) == HMLL_DTYPE_FLOAT32);
        REQUIRE(hmll_safetensors_dtype_from_str("F64", 4) == HMLL_DTYPE_FLOAT64);
        REQUIRE(hmll_safetensors_dtype_from_str("F4", 3) == HMLL_DTYPE_FLOAT4);
    }

    SECTION("float8 types") {
        REQUIRE(hmll_safetensors_dtype_from_str("F8_E4M3", 7) == HMLL_DTYPE_FLOAT8_E4M3);
        REQUIRE(hmll_safetensors_dtype_from_str("F8_E5M2", 7) == HMLL_DTYPE_FLOAT8_E5M2);
    }

    SECTION("float6 types") {
        REQUIRE(hmll_safetensors_dtype_from_str("F6_E2M3", 7) == HMLL_DTYPE_FLOAT6_E2M3);
        REQUIRE(hmll_safetensors_dtype_from_str("F6_E3M2", 7) == HMLL_DTYPE_FLOAT6_E3M2);
    }

    SECTION("complex types") {
        REQUIRE(hmll_safetensors_dtype_from_str("C64", 3) == HMLL_DTYPE_COMPLEX);
    }

    SECTION("signed integer types") {
        REQUIRE(hmll_safetensors_dtype_from_str("I8", 2) == HMLL_DTYPE_SIGNED_INT8);
        REQUIRE(hmll_safetensors_dtype_from_str("I16", 3) == HMLL_DTYPE_SIGNED_INT16);
        REQUIRE(hmll_safetensors_dtype_from_str("I32", 3) == HMLL_DTYPE_SIGNED_INT32);
        REQUIRE(hmll_safetensors_dtype_from_str("I64", 3) == HMLL_DTYPE_SIGNED_INT64);
    }

    SECTION("unsigned integer types") {
        REQUIRE(hmll_safetensors_dtype_from_str("U8", 2) == HMLL_DTYPE_UNSIGNED_INT8);
        REQUIRE(hmll_safetensors_dtype_from_str("U16", 3) == HMLL_DTYPE_UNSIGNED_INT16);
        REQUIRE(hmll_safetensors_dtype_from_str("U32", 3) == HMLL_DTYPE_UNSIGNED_INT32);
        REQUIRE(hmll_safetensors_dtype_from_str("U64", 3) == HMLL_DTYPE_UNSIGNED_INT64);
    }

    SECTION("unknown dtype") {
        REQUIRE(hmll_safetensors_dtype_from_str("INVALID", 7) == HMLL_DTYPE_UNKNOWN);
        REQUIRE(hmll_safetensors_dtype_from_str("", 0) == HMLL_DTYPE_UNKNOWN);
        REQUIRE(hmll_safetensors_dtype_from_str("FP64", 4) == HMLL_DTYPE_UNKNOWN);
    }

    SECTION("case sensitivity") {
        // safetensors format is case-sensitive
        REQUIRE(hmll_safetensors_dtype_from_str("bool", 4) == HMLL_DTYPE_UNKNOWN);
        REQUIRE(hmll_safetensors_dtype_from_str("bf16", 4) == HMLL_DTYPE_UNKNOWN);
        REQUIRE(hmll_safetensors_dtype_from_str("fp32", 4) == HMLL_DTYPE_UNKNOWN);
    }
}

TEST_CASE("safetensors path creation", "[safetensors]")
{
    SECTION("create valid path") {
        const auto base = "/path/to/model.safetensors";
        const auto file = "model-00002-of-00005.safetensors";
        auto result = hmll_safetensors_path_create(base, file);

        REQUIRE(result != nullptr);
        REQUIRE(strcmp(result, "/path/to/model-00002-of-00005.safetensors") == 0);

        free(result);
    }

    SECTION("handle null inputs") {
        REQUIRE(hmll_safetensors_path_create(nullptr, "file.safetensors") == nullptr);
        REQUIRE(hmll_safetensors_path_create("/path/model.safetensors", nullptr) == nullptr);
    }

    SECTION("handle path without directory") {
        const auto base = "model.safetensors";
        const auto file = "other.safetensors";
        auto result = hmll_safetensors_path_create(base, file);

        REQUIRE(result != nullptr);
        REQUIRE(strcmp(result, "other.safetensors") == 0);

        free(result);
    }
}

TEST_CASE("safetensors offset parsing", "[safetensors]")
{
    SECTION("parse valid offsets") {
        const auto json = "[100, 500]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_offsets(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.start == 100);
        REQUIRE(tensor.end == 500);

        yyjson_doc_free(doc);
    }

    SECTION("parse offsets with large values") {
        const auto json = "[1073741824, 2147483648]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_offsets(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.start == 1073741824);
        REQUIRE(tensor.end == 2147483648);

        yyjson_doc_free(doc);
    }

    SECTION("reject malformed offsets - not array") {
        const auto json = "\"not an array\"";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_offsets(root, &tensor);

        REQUIRE(hmll_check(err) == true);
        REQUIRE(err.code == HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);

        yyjson_doc_free(doc);
    }

    SECTION("reject malformed offsets - single element") {
        const auto json = "[100]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_offsets(root, &tensor);

        REQUIRE(hmll_check(err) == true);
        REQUIRE(err.code == HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);

        yyjson_doc_free(doc);
    }

    SECTION("reject malformed offsets - empty array") {
        const auto json = "[]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_offsets(root, &tensor);

        REQUIRE(hmll_check(err) == true);
        REQUIRE(err.code == HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);

        yyjson_doc_free(doc);
    }
}

TEST_CASE("safetensors shape parsing", "[safetensors]")
{
    SECTION("parse 1D shape") {
        const auto json = "[1024]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.rank == 1);
        REQUIRE(tensor.shape[0] == 1024);

        yyjson_doc_free(doc);
    }

    SECTION("parse 2D shape") {
        const auto json = "[32, 64]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.rank == 2);
        REQUIRE(tensor.shape[0] == 32);
        REQUIRE(tensor.shape[1] == 64);

        yyjson_doc_free(doc);
    }

    SECTION("parse 4D shape") {
        const auto json = "[2, 3, 224, 224]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.rank == 4);
        REQUIRE(tensor.shape[0] == 2);
        REQUIRE(tensor.shape[1] == 3);
        REQUIRE(tensor.shape[2] == 224);
        REQUIRE(tensor.shape[3] == 224);

        yyjson_doc_free(doc);
    }

    SECTION("parse scalar (empty shape)") {
        const auto json = "[]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.rank == 0);

        yyjson_doc_free(doc);
    }

    SECTION("reject malformed shape - not array") {
        const auto json = "42";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == true);
        REQUIRE(err.code == HMLL_ERR_SAFETENSORS_JSON_MALFORMED_HEADER);

        yyjson_doc_free(doc);
    }

    SECTION("parse large shape values") {
        const auto json = "[4096, 4096]";
        auto doc = yyjson_read(json, strlen(json), 0);
        auto root = yyjson_doc_get_root(doc);

        hmll_tensor_specs_t tensor = {};
        hmll_error_t err = hmll_safetensors_header_parse_shape(root, &tensor);

        REQUIRE(hmll_check(err) == false);
        REQUIRE(tensor.rank == 2);
        REQUIRE(tensor.shape[0] == 4096);
        REQUIRE(tensor.shape[1] == 4096);

        yyjson_doc_free(doc);
    }
}
