#include "tinytensor/tensor.hpp"
#include <iostream>
#include <vector>

int main() {
    tt::Tensor t;
    auto r = tt::Tensor::CreateFloat32(tt::Shape{ {10} }, tt::Layout::AoS, t);
    if (!r.ok()) { std::cerr << "Create failed: " << tt::ToString(r.status) << "\n"; return 1; }

    std::vector<float> src(10);
    for (int i = 0; i < 10; ++i) src[i] = float(i) * 0.5f;

    r = t.copy_from(src.data(), src.size() * sizeof(float));
    if (!r.ok()) { std::cerr << "copy_from failed\n"; return 1; }

    std::vector<float> dst(10, -1.0f);
    r = t.copy_to(dst.data(), dst.size() * sizeof(float));
    if (!r.ok()) { std::cerr << "copy_to failed\n"; return 1; }

    for (auto x : dst) std::cout << x << " ";
    std::cout << "\n";
}
