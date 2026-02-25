# TinyTensor

TinyTensor 是一个基于 **C++20** 构建的轻量级高性能 Tensor 库，旨在探索和实现深度学习框架底层的核心概念。

该项目通过三个核心挑战，从零构建了一个具备 N 维数据管理和并行计算能力的 Tensor 系统。

## ✨ 核心特性

*   **N 维 Tensor 支持**
    *   支持任意维度的 `Shape` 定义。
    *   实现了通用的 N 维 `Strides`（步幅）计算逻辑，能够正确处理多维数组的索引。

*   **灵活的内存布局 (Memory Layout)**
    *   **AoS (Array of Structures)**：支持行优先（Row-Major）布局。
    *   **SoA (Structure of Arrays)**：支持列优先（Column-Major）布局。
    *   通过统一的步幅算法适配不同的数据排布方式。

*   **并行计算能力**
    *   集成了自定义的 `ThreadPool` 线程池。
    *   实现了 `ParallelAdd`，利用多线程将计算任务分块执行，并通过 `std::atomic` 实现高效的线程同步。

*   **底层优化**
    *   基于 `AlignedBuffer` 实现内存对齐（64字节对齐），优化 SIMD 访问性能。
    *   使用现代 CMake (Ninja Generator) 进行构建管理。

## 🛠️ 技术栈

*   **语言**: C++20
*   **构建工具**: CMake (Minimum 3.20)
*   **关键技术**: `std::atomic`, `std::thread`, `std::accumulate`, PImpl 模式
