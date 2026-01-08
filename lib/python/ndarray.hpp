#ifndef PYHMLL_NDARRAY_HPP
#define PYHMLL_NDARRAY_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


constexpr int kDLPACK_DEVICE_CPU  = 1;
constexpr int kDLPACK_DEVICE_CUDA = 2;

// DLPack Dtype Codes
constexpr nb::dlpack::dtype kBOOL_DTYPE         = {static_cast<uint8_t>(nb::dlpack::dtype_code::Bool), 8, 1};
constexpr nb::dlpack::dtype kBF16_DTYPE         = {static_cast<uint8_t>(nb::dlpack::dtype_code::Bfloat), 16, 1};
constexpr nb::dlpack::dtype kCOMPLEX64_DTYPE    = {static_cast<uint8_t>(nb::dlpack::dtype_code::Complex), 64, 1};
constexpr nb::dlpack::dtype kCOMPLEX128_DTYPE   = {static_cast<uint8_t>(nb::dlpack::dtype_code::Complex), 128, 1};
constexpr nb::dlpack::dtype kFLOAT4_DTYPE       = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float4_E2M1FN), 4, 1};
constexpr nb::dlpack::dtype kFLOAT6_E2M3_DTYPE  = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float6_E2M3FN), 6, 1};
constexpr nb::dlpack::dtype kFLOAT6_E3M2_DTYPE  = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float6_E3M2FN), 6, 1};
constexpr nb::dlpack::dtype kFLOAT8_E5M2_DTYPE  = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float8_E5M2), 8, 1};
constexpr nb::dlpack::dtype kFLOAT8_E4M3_DTYPE  = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float8_E4M3), 8, 1};
constexpr nb::dlpack::dtype kFLOAT8_E8M0_DTYPE  = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float8_E8M0FNU), 8, 1};
constexpr nb::dlpack::dtype kF16_DTYPE          = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
constexpr nb::dlpack::dtype kF32_DTYPE          = {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 32, 1};
constexpr nb::dlpack::dtype kINT4_DTYPE         = {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 4, 1};
constexpr nb::dlpack::dtype kINT8_DTYPE         = {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 8, 1};
constexpr nb::dlpack::dtype kINT16_DTYPE        = {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 16, 1};
constexpr nb::dlpack::dtype kINT32_DTYPE        = {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 32, 1};
constexpr nb::dlpack::dtype kINT64_DTYPE        = {static_cast<uint8_t>(nb::dlpack::dtype_code::Int), 64, 1};
constexpr nb::dlpack::dtype kUINT4_DTYPE        = {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 4, 1};
constexpr nb::dlpack::dtype kUINT8_DTYPE        = {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 8, 1};
constexpr nb::dlpack::dtype kUINT16_DTYPE       = {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 16, 1};
constexpr nb::dlpack::dtype kUINT32_DTYPE       = {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 32, 1};
constexpr nb::dlpack::dtype kUINT64_DTYPE       = {static_cast<uint8_t>(nb::dlpack::dtype_code::UInt), 64, 1};


static constexpr nb::dlpack::dtype hmll_dtype_to_dlpack(const hmll_dtype_t dtype)
{
    switch (dtype)
    {
    case HMLL_DTYPE_BOOL:
        return kBOOL_DTYPE;
    case HMLL_DTYPE_BFLOAT16:
        return kBF16_DTYPE;
    case HMLL_DTYPE_COMPLEX:
        return kCOMPLEX64_DTYPE; // Default to complex64
    case HMLL_DTYPE_FLOAT4:
        return kFLOAT4_DTYPE;
    case HMLL_DTYPE_FLOAT6_E2M3:
        return kFLOAT6_E2M3_DTYPE;
    case HMLL_DTYPE_FLOAT6_E3M2:
        return kFLOAT6_E3M2_DTYPE;
    case HMLL_DTYPE_FLOAT8_E5M2:
        return kFLOAT8_E5M2_DTYPE;
    case HMLL_DTYPE_FLOAT8_E4M3:
        return kFLOAT8_E4M3_DTYPE;
    case HMLL_DTYPE_FLOAT8_E8M0:
        return kFLOAT8_E8M0_DTYPE;
    case HMLL_DTYPE_FLOAT16:
        return kF16_DTYPE;
    case HMLL_DTYPE_FLOAT32:
        return kF32_DTYPE;
    case HMLL_DTYPE_SIGNED_INT4:
        return kINT4_DTYPE;
    case HMLL_DTYPE_SIGNED_INT8:
        return kINT8_DTYPE;
    case HMLL_DTYPE_SIGNED_INT16:
        return kINT16_DTYPE;
    case HMLL_DTYPE_SIGNED_INT32:
        return kINT32_DTYPE;
    case HMLL_DTYPE_SIGNED_INT64:
        return kINT64_DTYPE;
    case HMLL_DTYPE_UNSIGNED_INT4:
        return kUINT4_DTYPE;
    case HMLL_DTYPE_UNSIGNED_INT8:
        return kUINT8_DTYPE;
    case HMLL_DTYPE_UNSIGNED_INT16:
        return kUINT16_DTYPE;
    case HMLL_DTYPE_UNSIGNED_INT32:
        return kUINT32_DTYPE;
    case HMLL_DTYPE_UNSIGNED_INT64:
        return kUINT64_DTYPE;
    default:
        return kUINT8_DTYPE; // Fallback to uint8
    }
}


static nb::ndarray<nb::ndim<1>, nb::c_contig> hmll_to_ndarray(
    const hmll_range_t range,
    const hmll_iobuf_t& buffer,
    const hmll_range_t offsets,
    const hmll_dtype_t dtype,
    const nb::object& owner
) {

    int32_t device_type, device_id;
    switch (buffer.device)
    {
    case HMLL_DEVICE_CUDA:
        device_type = kDLPACK_DEVICE_CUDA;
        device_id = 0;
        break;
    default:
        device_type = kDLPACK_DEVICE_CPU;
        device_id = 0;
    }

    const auto dtype_ = hmll_dtype_to_dlpack(dtype);
    const auto numel_ = (range.end - range.start) / (dtype_.bits / 8);
    return nb::ndarray<nb::ndim<1>, nb::c_contig> (
        static_cast<unsigned char*>(buffer.ptr) + offsets.start,
        {numel_},
        owner,
        {},
        dtype_,
        device_type,
        device_id,
        'C'
    );
}

#endif // PYHMLL_NDARRAY_HPP
