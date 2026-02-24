#pragma once
#include <cstddef>
#include <functional>

namespace tt {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t n = 0); // n=0 则取 hardware_concurrency
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void enqueue(std::function<void()> job);

private:
    // PImpl 隐藏实现细节，减少头文件依赖
    struct Impl;
    Impl* impl_;  // impl_是一个内部指向实现的指针
};

} // namespace tt
