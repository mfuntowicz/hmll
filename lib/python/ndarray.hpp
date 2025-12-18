#ifndef PYHMLL_NDARRAY_HPP
#define PYHMLL_NDARRAY_HPP

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;


constexpr int kDLPACK_DEVICE_CPU = 1;

// DLPack Dtype Codes
constexpr nb::dlpack::dtype kBF16_DTYPE = {4, 16, 1};
constexpr nb::dlpack::dtype kF16_DTYPE  = {2, 16, 1};
constexpr nb::dlpack::dtype kF32_DTYPE  = {2, 32, 1};

static nb::ndarray<> hmll_to_ndarray(
    const hmll_tensor_specs_t& specs,
    const hmll_device_buffer_t& buffer,
    const hmll_fetch_range_t offsets,
    const nb::object& owner
) {
    // 1. Resolve Dtype (Runtime)
    nb::dlpack::dtype dt;
    switch (specs.dtype) {
    case HMLL_DTYPE_BFLOAT16: dt = kBF16_DTYPE; break;
    case HMLL_DTYPE_FLOAT16:  dt = kF16_DTYPE;  break;
    case HMLL_DTYPE_FLOAT32:  dt = kF32_DTYPE;  break;
    default: throw std::runtime_error("Unknown dtype");
    }

    // 2. Resolve Device (Runtime)
    // You need to map your internal hmll device enum to DLPack codes
    int32_t device_type = kDLPACK_DEVICE_CPU;
    int32_t device_id = 0;

    // Example mapping logic (Adjust to your actual hmll struct!)
    // if (buffer.device == HMLL_DEVICE_CUDA) {
    //     device_type = kDLCUDA;
    //     device_id = buffer.device_id;
    // } else if (buffer.device == HMLL_DEVICE_ROCM) {
    //     device_type = kDLROCM;
    //     device_id = buffer.device_id;
    // }
    // else CPU (default)

    // 3. Construct with explicit Device info
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
