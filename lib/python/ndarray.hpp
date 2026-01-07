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

static nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> hmll_to_ndarray(
    const hmll_range_t range,
    const hmll_iobuf_t& buffer,
    const hmll_range_t offsets,
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

    return nb::ndarray<unsigned char, nb::ndim<1>, nb::c_contig> (
        static_cast<unsigned char*>(buffer.ptr) + offsets.start,
        {range.end - range.start},
        owner,
        {},
        nb::dtype<unsigned char>(),
        device_type,
        device_id,
        'C'
    );
}

#endif // PYHMLL_NDARRAY_HPP
