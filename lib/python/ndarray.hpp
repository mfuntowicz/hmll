#ifndef PYHMLL_NDARRAY_HPP
#define PYHMLL_NDARRAY_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


constexpr int kDLPACK_DEVICE_CPU  = 1;
constexpr int kDLPACK_DEVICE_CUDA = 2;

// DLPack Dtype Codes
constexpr nb::dlpack::dtype kBF16_DTYPE = {4, 16, 1};
constexpr nb::dlpack::dtype kF16_DTYPE  = {2, 16, 1};
constexpr nb::dlpack::dtype kF32_DTYPE  = {2, 32, 1};

static nb::ndarray<> hmll_to_ndarray(
    const hmll_tensor_specs_t& specs,
    const hmll_device_buffer_t& buffer,
    const hmll_range_t offsets,
    const nb::object& owner
) {
    nb::dlpack::dtype dt;
    switch (specs.dtype) {
    case HMLL_DTYPE_BFLOAT16: dt = kBF16_DTYPE; break;
    case HMLL_DTYPE_FLOAT16:  dt = kF16_DTYPE;  break;
    case HMLL_DTYPE_FLOAT32:  dt = kF32_DTYPE;  break;
    default: throw std::runtime_error("Unknown dtype");
    }

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

    return nb::ndarray{
        static_cast<char*>(buffer.ptr) + offsets.start,
        specs.rank,
        specs.shape,
        owner,
        nullptr,
        dt,
        device_type,
        device_id,
        'C'
    };
}

#endif // PYHMLL_NDARRAY_HPP
