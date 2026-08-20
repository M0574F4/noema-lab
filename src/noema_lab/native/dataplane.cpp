#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace py = pybind11;

template <typename T>
py::array_t<T> make_1d(std::size_t n) {
    return py::array_t<T>({static_cast<py::ssize_t>(n)});
}

py::array_t<uint8_t> packbits_u8(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits) {
    auto b = bits.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(b.shape(0));
    auto out = make_1d<uint8_t>((n + 7u) / 8u);
    auto o = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < o.shape(0); ++i) {
        o(i) = 0;
    }
    for (std::size_t i = 0; i < n; ++i) {
        o(static_cast<py::ssize_t>(i >> 3u)) |= static_cast<uint8_t>((b(static_cast<py::ssize_t>(i)) & 1u) << (7u - (i & 7u)));
    }
    return out;
}

py::array_t<uint8_t> unpackbits_u8(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bytes,
    std::size_t bit_count
) {
    auto data = bytes.unchecked<1>();
    if (bit_count > static_cast<std::size_t>(data.shape(0)) * 8u) {
        throw std::runtime_error("bit_count exceeds byte payload");
    }
    auto out = make_1d<uint8_t>(bit_count);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < bit_count; ++i) {
        o(static_cast<py::ssize_t>(i)) = static_cast<uint8_t>((data(static_cast<py::ssize_t>(i >> 3u)) >> (7u - (i & 7u))) & 1u);
    }
    return out;
}

py::array_t<uint8_t> indices_to_bits_i64(
    py::array_t<int64_t, py::array::c_style | py::array::forcecast> indices,
    int width
) {
    if (width <= 0 || width > 31) {
        throw std::runtime_error("width must be in [1, 31]");
    }
    auto x = indices.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(x.shape(0));
    auto out = make_1d<uint8_t>(n * static_cast<std::size_t>(width));
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t value = static_cast<uint32_t>(x(static_cast<py::ssize_t>(i)));
        const std::size_t base = i * static_cast<std::size_t>(width);
        for (int j = 0; j < width; ++j) {
            o(static_cast<py::ssize_t>(base + static_cast<std::size_t>(j))) =
                static_cast<uint8_t>((value >> (width - 1 - j)) & 1u);
        }
    }
    return out;
}

py::array_t<int64_t> bits_to_indices_i64(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits,
    int width,
    std::size_t count,
    int64_t codebook_size,
    bool clamp
) {
    if (width <= 0 || width > 31) {
        throw std::runtime_error("width must be in [1, 31]");
    }
    auto b = bits.unchecked<1>();
    if (static_cast<std::size_t>(b.shape(0)) < count * static_cast<std::size_t>(width)) {
        throw std::runtime_error("not enough bits");
    }
    auto out = make_1d<int64_t>(count);
    auto o = out.mutable_unchecked<1>();
    const int64_t max_value = std::max<int64_t>(codebook_size - 1, 0);
    for (std::size_t i = 0; i < count; ++i) {
        uint32_t value = 0;
        const std::size_t base = i * static_cast<std::size_t>(width);
        for (int j = 0; j < width; ++j) {
            value = (value << 1u) | static_cast<uint32_t>(b(static_cast<py::ssize_t>(base + static_cast<std::size_t>(j))) & 1u);
        }
        int64_t decoded = static_cast<int64_t>(value);
        if (clamp) {
            decoded = std::min<int64_t>(std::max<int64_t>(decoded, 0), max_value);
        } else if (codebook_size > 0) {
            decoded %= codebook_size;
        }
        o(static_cast<py::ssize_t>(i)) = decoded;
    }
    return out;
}

py::array_t<uint8_t> repetition_encode(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits,
    int factor
) {
    if (factor <= 0) {
        throw std::runtime_error("factor must be positive");
    }
    auto b = bits.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(b.shape(0));
    auto out = make_1d<uint8_t>(n * static_cast<std::size_t>(factor));
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t value = static_cast<uint8_t>(b(static_cast<py::ssize_t>(i)) & 1u);
        const std::size_t base = i * static_cast<std::size_t>(factor);
        for (int j = 0; j < factor; ++j) {
            o(static_cast<py::ssize_t>(base + static_cast<std::size_t>(j))) = value;
        }
    }
    return out;
}

py::array_t<uint8_t> repetition_decode(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits,
    int factor,
    std::size_t payload_count
) {
    if (factor <= 0) {
        throw std::runtime_error("factor must be positive");
    }
    auto b = bits.unchecked<1>();
    const std::size_t groups = static_cast<std::size_t>(b.shape(0)) / static_cast<std::size_t>(factor);
    const std::size_t out_count = std::min<std::size_t>(groups, payload_count);
    auto out = make_1d<uint8_t>(out_count);
    auto o = out.mutable_unchecked<1>();
    const int threshold = (factor + 1) / 2;
    for (std::size_t i = 0; i < out_count; ++i) {
        int ones = 0;
        const std::size_t base = i * static_cast<std::size_t>(factor);
        for (int j = 0; j < factor; ++j) {
            ones += static_cast<int>(b(static_cast<py::ssize_t>(base + static_cast<std::size_t>(j))) & 1u);
        }
        o(static_cast<py::ssize_t>(i)) = static_cast<uint8_t>(ones >= threshold);
    }
    return out;
}

int64_t ber_count(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> b
) {
    auto x = a.unchecked<1>();
    auto y = b.unchecked<1>();
    const std::size_t n = std::min<std::size_t>(static_cast<std::size_t>(x.shape(0)), static_cast<std::size_t>(y.shape(0)));
    int64_t errors = 0;
    for (std::size_t i = 0; i < n; ++i) {
        errors += static_cast<int64_t>((x(static_cast<py::ssize_t>(i)) & 1u) != (y(static_cast<py::ssize_t>(i)) & 1u));
    }
    return errors;
}

py::array_t<std::complex<float>> qpsk_modulate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits
) {
    auto b = bits.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(b.shape(0));
    const std::size_t symbol_count = (n + 1u) / 2u;
    auto out = make_1d<std::complex<float>>(symbol_count);
    auto o = out.mutable_unchecked<1>();
    const float scale = static_cast<float>(1.0 / std::sqrt(2.0));
    for (std::size_t i = 0; i < symbol_count; ++i) {
        const std::size_t bit_i = i * 2u;
        const uint8_t bit0 = bit_i < n ? static_cast<uint8_t>(b(static_cast<py::ssize_t>(bit_i)) & 1u) : 0u;
        const uint8_t bit1 = bit_i + 1u < n ? static_cast<uint8_t>(b(static_cast<py::ssize_t>(bit_i + 1u)) & 1u) : 0u;
        o(static_cast<py::ssize_t>(i)) = std::complex<float>(bit0 ? -scale : scale, bit1 ? -scale : scale);
    }
    return out;
}

py::array_t<uint8_t> qpsk_demodulate(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> symbols
) {
    auto s = symbols.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(s.shape(0));
    auto out = make_1d<uint8_t>(n * 2u);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const auto value = s(static_cast<py::ssize_t>(i));
        o(static_cast<py::ssize_t>(2u * i)) = static_cast<uint8_t>(value.real() < 0.0f);
        o(static_cast<py::ssize_t>(2u * i + 1u)) = static_cast<uint8_t>(value.imag() < 0.0f);
    }
    return out;
}

py::array_t<std::complex<float>> bpsk_modulate(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits
) {
    auto b = bits.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(b.shape(0));
    auto out = make_1d<std::complex<float>>(n);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        o(static_cast<py::ssize_t>(i)) = std::complex<float>((b(static_cast<py::ssize_t>(i)) & 1u) ? -1.0f : 1.0f, 0.0f);
    }
    return out;
}

py::array_t<uint8_t> bpsk_demodulate(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> symbols
) {
    auto s = symbols.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(s.shape(0));
    auto out = make_1d<uint8_t>(n);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        o(static_cast<py::ssize_t>(i)) = static_cast<uint8_t>(s(static_cast<py::ssize_t>(i)).real() < 0.0f);
    }
    return out;
}

py::array_t<std::complex<float>> awgn_apply(
    py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> symbols,
    py::array_t<float, py::array::c_style | py::array::forcecast> noise_real,
    py::array_t<float, py::array::c_style | py::array::forcecast> noise_imag,
    float scale
) {
    auto s = symbols.unchecked<1>();
    auto nr = noise_real.unchecked<1>();
    auto ni = noise_imag.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(s.shape(0));
    if (static_cast<std::size_t>(nr.shape(0)) < n || static_cast<std::size_t>(ni.shape(0)) < n) {
        throw std::runtime_error("noise arrays are too short");
    }
    auto out = make_1d<std::complex<float>>(n);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        o(static_cast<py::ssize_t>(i)) =
            s(static_cast<py::ssize_t>(i)) + std::complex<float>(scale * nr(static_cast<py::ssize_t>(i)), scale * ni(static_cast<py::ssize_t>(i)));
    }
    return out;
}

py::array_t<uint8_t> require_canonical_bits(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits
) {
    auto b = bits.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(b.shape(0));
    auto out = make_1d<uint8_t>(n);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        const uint8_t value = b(static_cast<py::ssize_t>(i));
        if (value != 0u && value != 1u) {
            throw std::runtime_error("non-binary bit value");
        }
        o(static_cast<py::ssize_t>(i)) = value;
    }
    return out;
}

py::array_t<float> image_to_nchw(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> images
) {
    auto x = images.unchecked<4>();
    const py::ssize_t n = x.shape(0);
    const py::ssize_t h = x.shape(1);
    const py::ssize_t w = x.shape(2);
    const py::ssize_t c = x.shape(3);
    auto out = py::array_t<float>({n, c, h, w});
    auto o = out.mutable_unchecked<4>();
    for (py::ssize_t b = 0; b < n; ++b) {
        for (py::ssize_t yy = 0; yy < h; ++yy) {
            for (py::ssize_t xx = 0; xx < w; ++xx) {
                for (py::ssize_t ch = 0; ch < c; ++ch) {
                    o(b, ch, yy, xx) = static_cast<float>(x(b, yy, xx, ch)) / 255.0f;
                }
            }
        }
    }
    return out;
}

py::array_t<uint8_t> nchw_to_image(
    py::array_t<float, py::array::c_style | py::array::forcecast> tensor
) {
    auto x = tensor.unchecked<4>();
    const py::ssize_t n = x.shape(0);
    const py::ssize_t c = x.shape(1);
    const py::ssize_t h = x.shape(2);
    const py::ssize_t w = x.shape(3);
    auto out = py::array_t<uint8_t>({n, h, w, c});
    auto o = out.mutable_unchecked<4>();
    for (py::ssize_t b = 0; b < n; ++b) {
        for (py::ssize_t yy = 0; yy < h; ++yy) {
            for (py::ssize_t xx = 0; xx < w; ++xx) {
                for (py::ssize_t ch = 0; ch < c; ++ch) {
                    float value = x(b, ch, yy, xx);
                    value = std::min<float>(std::max<float>(value, 0.0f), 1.0f);
                    o(b, yy, xx, ch) = static_cast<uint8_t>(std::nearbyint(value * 255.0f));
                }
            }
        }
    }
    return out;
}

py::array_t<uint8_t> float32_to_bits(
    py::array_t<float, py::array::c_style | py::array::forcecast> values
) {
    auto x = values.unchecked<1>();
    const std::size_t n = static_cast<std::size_t>(x.shape(0));
    auto out = make_1d<uint8_t>(n * sizeof(float) * 8u);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t word = 0;
        const float value = x(static_cast<py::ssize_t>(i));
        std::memcpy(&word, &value, sizeof(float));
        const std::size_t base = i * sizeof(float) * 8u;
        for (std::size_t j = 0; j < sizeof(float); ++j) {
            const uint8_t byte = static_cast<uint8_t>((word >> (j * 8u)) & 0xffu);
            for (std::size_t k = 0; k < 8u; ++k) {
                o(static_cast<py::ssize_t>(base + j * 8u + k)) = static_cast<uint8_t>((byte >> (7u - k)) & 1u);
            }
        }
    }
    return out;
}

py::array_t<float> bits_to_float32(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bits
) {
    auto b = bits.unchecked<1>();
    const std::size_t bit_count = static_cast<std::size_t>(b.shape(0));
    const std::size_t n = bit_count / (sizeof(float) * 8u);
    auto out = make_1d<float>(n);
    auto o = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < n; ++i) {
        uint32_t word = 0;
        const std::size_t base = i * sizeof(float) * 8u;
        for (std::size_t j = 0; j < sizeof(float); ++j) {
            uint8_t byte = 0;
            for (std::size_t k = 0; k < 8u; ++k) {
                byte |= static_cast<uint8_t>((b(static_cast<py::ssize_t>(base + j * 8u + k)) & 1u) << (7u - k));
            }
            word |= static_cast<uint32_t>(byte) << (j * 8u);
        }
        float value = 0.0f;
        std::memcpy(&value, &word, sizeof(float));
        o(static_cast<py::ssize_t>(i)) = value;
    }
    return out;
}

PYBIND11_MODULE(_native_dataplane, m) {
    m.def("packbits_u8", &packbits_u8);
    m.def("unpackbits_u8", &unpackbits_u8);
    m.def("indices_to_bits_i64", &indices_to_bits_i64);
    m.def("bits_to_indices_i64", &bits_to_indices_i64);
    m.def("repetition_encode", &repetition_encode);
    m.def("repetition_decode", &repetition_decode);
    m.def("ber_count", &ber_count);
    m.def("qpsk_modulate", &qpsk_modulate);
    m.def("qpsk_demodulate", &qpsk_demodulate);
    m.def("bpsk_modulate", &bpsk_modulate);
    m.def("bpsk_demodulate", &bpsk_demodulate);
    m.def("awgn_apply", &awgn_apply);
    m.def("require_canonical_bits", &require_canonical_bits);
    m.def("image_to_nchw", &image_to_nchw);
    m.def("nchw_to_image", &nchw_to_image);
    m.def("float32_to_bits", &float32_to_bits);
    m.def("bits_to_float32", &bits_to_float32);
}
