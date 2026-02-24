#include "tinytensor/tensor.hpp"
#include "tinytensor/thread_pool.hpp"
#include <iostream>
#include <vector>
#include <atomic>
#include <thread>

// 简易异步：把 copy+compute+copy 三步用线程池并联

int main() {
    tt::Tensor x;
    tt::Tensor y;
    auto rx = tt::Tensor::CreateFloat32(tt::Shape{ {100000} }, tt::Layout::AoS, x);
    auto ry = tt::Tensor::CreateFloat32(tt::Shape{ {100000} }, tt::Layout::AoS, y);
    if (!rx.ok() || !ry.ok()) { std::cerr << "Create failed\n"; return 1; }

    std::vector<float> host_in(x.bytes() / sizeof(float));
    for (size_t i = 0; i < host_in.size(); ++i) 
        host_in[i] = float(i % 100);

    std::vector<float> host_out(y.bytes() / sizeof(float), 0.0f);

    // 线程池（模拟 CUDA stream）
    tt::ThreadPool pool(4);
    std::atomic<bool> done{ false };

    pool.enqueue([&] {
        //std::this_thread::sleep_for(std::chrono::seconds(1));
        x.copy_from(host_in.data(), x.bytes());
        });
    pool.enqueue([&] {
        // "compute"：在 x 上做简单计算，写入 y（当作 kernel）
        float* xd = x.data();
        float* yd = y.data();
        size_t n = x.bytes() / sizeof(float);
        for (size_t i = 0; i < n; ++i) yd[i] = xd[i] * 2.0f + 1.0f;
        });
    pool.enqueue([&] {
        y.copy_to(host_out.data(), y.bytes());
        done = true;
        });

    while (!done) std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // 检验
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        float expect = host_in[i] * 2.0f + 1.0f;
        if (host_out[i] != expect) ok = false;
        std::cout << host_out[i] << " ";
    }
    std::cout << "\nResult: " << (ok ? "OK" : "Mismatch") << "\n";
}
