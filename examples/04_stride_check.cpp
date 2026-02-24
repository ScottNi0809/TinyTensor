#include "tinytensor/tensor.hpp"
#include <iostream>
#include <vector>

void print_strides(const std::string& name, const tt::Tensor& t) {
    std::cout << "Tensor: " << name << " [ ";
    for (auto d : t.shape().dims) std::cout << d << " ";
    std::cout << "] Layout=" << (t.layout() == tt::Layout::AoS ? "AoS" : "SoA") << "\n";
    
    std::cout << "  Calculated Strides (bytes): [ ";
    for (size_t i = 0; i < t.shape().dims.size(); ++i) {
        std::cout << t.stride(i) << " ";
    }
    std::cout << "]\n";
    std::cout << "  Expected Strides (bytes):   [ ";

    // ???????
    std::vector<size_t> expected(t.shape().dims.size());
    if (t.layout() == tt::Layout::AoS) {
        // AoS: Row Major - Right Most = 4
        expected.back() = 4; // float
        for (int i = (int)expected.size() - 2; i >= 0; --i) {
            expected[i] = expected[i+1] * t.shape().dims[i+1];
        }
    } else {
        // SoA: Col Major - Left Most = 4
        expected[0] = 4; // float
        for (size_t i = 1; i < expected.size(); ++i) {
            expected[i] = expected[i-1] * t.shape().dims[i-1];
        }
    }

    for (auto s : expected) std::cout << s << " ";
    std::cout << "]\n\n";
}

int main() {
    tt::Tensor t3d_aos, t3d_soa;
    tt::Tensor t4d_aos, t4d_soa;
    
    // Case 1: 3D [2, 3, 4] AoS
    // Stride[2] = 4
    // Stride[1] = 4 * 4 = 16
    // Stride[0] = 3 * 16 = 48
    tt::Tensor::CreateFloat32(tt::Shape{{2, 3, 4}}, tt::Layout::AoS, t3d_aos);
    print_strides("3D AoS", t3d_aos);

    // Case 2: 3D [2, 3, 4] SoA
    // Stride[0] = 4
    // Stride[1] = 2 * 4 = 8
    // Stride[2] = 3 * 8 = 24
    tt::Tensor::CreateFloat32(tt::Shape{{2, 3, 4}}, tt::Layout::SoA, t3d_soa);
    print_strides("3D SoA", t3d_soa);

    // Case 3: 4D [2, 3, 4, 5] AoS (Image Batch NCHW)
    // Stride[3] = 4
    // Stride[2] = 5 * 4 = 20
    // Stride[1] = 4 * 20 = 80
    // Stride[0] = 3 * 80 = 240
    tt::Tensor::CreateFloat32(tt::Shape{{2, 3, 4, 5}}, tt::Layout::AoS, t4d_aos);
    print_strides("4D AoS", t4d_aos);

    return 0;
}
