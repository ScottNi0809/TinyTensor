#include "tinytensor/tensor.hpp"
#include <iostream>
#include <vector>

// 假设 N=4, C=3，把 AoS: [n][c] 连续，转换为 SoA: 每个 c 连续
int main() {
    const std::size_t N = 4, C = 3;

    tt::Tensor aos;
    tt::Tensor soa;
    auto r1 = tt::Tensor::CreateFloat32(tt::Shape{ {N, C} }, tt::Layout::AoS, aos);
    auto r2 = tt::Tensor::CreateFloat32(tt::Shape{ {N, C} }, tt::Layout::SoA, soa);
    if (!r1.ok() || !r2.ok()) { std::cerr << "Create failed\n"; return 1; }

    // 构造 AoS 数据: row-major [n][c]
    std::vector<float> tmp(N * C);
    for (std::size_t n = 0; n < N; ++n)
        for (std::size_t c = 0; c < C; ++c)
            tmp[n * C + c] = float(n * 10 + c); // 可视化数据

    aos.copy_from(tmp.data(), tmp.size() * sizeof(float));

    // 转换到 SoA
    // SoA 每个通道连续：buf = [c0(n=0..N-1), c1(...), c2(...)]
    std::vector<float> soa_data(N * C);
    for (std::size_t c = 0; c < C; ++c) {
        for (std::size_t n = 0; n < N; ++n) {
            soa_data[c * N + n] = tmp[n * C + c];
        }
    }
    soa.copy_from(soa_data.data(), soa_data.size() * sizeof(float));

    // 打印两者前 12 个数对比
    std::cout << "AoS: ";
    for (int i = 0; i < 12; ++i) std::cout << aos.data()[i] << " ";
    std::cout << "\nSoA: ";
    for (int i = 0; i < 12; ++i) std::cout << soa.data()[i] << " ";
    std::cout << "\n";
}
