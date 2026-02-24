#include "tinytensor/tensor.hpp"
#include "tinytensor/thread_pool.hpp" // 需要包含 ThreadPool 的定义
#include <thread> // for std::this_thread::yield
#include <vector>
#include <algorithm> // for std::min
#include <numeric>
#include <atomic>

namespace tt {

    static std::size_t product(const std::vector<std::size_t>& v) {
        return std::accumulate(v.begin(), v.end(), std::size_t{ 1 }, std::multiplies<>{});
    }

    // 根据你要求的形状（几乘几），计算出需要多少内存、怎么排布数据，并真正向系统申请这块内存。
    // 挑战二：实现通用的N维Tensor，支持任意维度
    Result Tensor::CreateFloat32(Shape shape, Layout layout, Tensor& out) {
        if (shape.dims.empty()) 
            return Result::Error(Status::InvalidArgument, "dims empty");

        out.shape_ = shape;
        out.layout_ = layout;

        // 计算字节数与步幅
        std::size_t total_elems = product(shape.dims);
		std::size_t bytes = total_elems * sizeof(float);  // 计算总字节数

        auto res = out.buf_.allocate(bytes, /*alignment=*/64);
        if (!res.ok()) 
            return res;

        // 初始化数据为 0
        std::memset(out.buf_.data(), 0, out.buf_.size());

        // 挑战二：实现通用的 N 维 Strides 计算
        if (layout == Layout::AoS) {
            // AoS: Row-Major (C-Style, 最右原则)
            // 最后一个维度步长是 1 个 float
            out.strides_.back() = sizeof(float);

            // 从倒数第二个开始往前推
            // Stride[i] = Stride[i+1] * Dim[i+1]
            // 注意用 int 以避免 size_t 下溢问题
            for (int i = (int)shape.dims.size() - 2; i >= 0; --i) {
                out.strides_[i] = out.strides_[i + 1] * shape.dims[i + 1];
            }
        }
        else {
            // SoA: Column-Major (Fortran-Style, 最左原则)
            // 第一个维度步长是 1 个 float
            out.strides_[0] = sizeof(float);

            // 从第二个开始往后推
            // Stride[i] = Stride[i-1] * Dim[i-1]
            for (size_t i = 1; i < shape.dims.size(); ++i) {
                out.strides_[i] = out.strides_[i - 1] * shape.dims[i - 1];
            }
        }
        
        return Result::OK();
    }

    Result Tensor::copy_from(const void* src, std::size_t bytes, std::size_t offset) {
        if (offset + bytes > buf_.size())
            return Result::Error(Status::CopyFailed, "copy_from overflow");
        std::memcpy(static_cast<char*>(buf_.data()) + offset, src, bytes);
        return Result::OK();
    }

    Result Tensor::copy_to(void* dst, std::size_t bytes, std::size_t offset) const {
        if (offset + bytes > buf_.size())
            return Result::Error(Status::CopyFailed, "copy_to overflow");
        std::memcpy(dst, static_cast<const char*>(buf_.data()) + offset, bytes);
        return Result::OK();
    }

    // 挑战三：实现真正的 ParallelAdd
    // 注意：这个函数最好是静态成员函数或者是自由函数，因为它操作三个 Tensor
    void Tensor::ParallelAdd(tt::ThreadPool& pool, const tt::Tensor& A, const tt::Tensor& B, tt::Tensor& C) {
        // 1. 简单校验
        if (A.bytes() != B.bytes() || A.bytes() != C.bytes()) {
            // 在产生环境应该返回 Error，这里简化处理直接 return
            return; 
        }

        // 1. 拿到原始指针（为了性能，不要在循环里调 .data()）
        const float* a_ptr = A.data();
        const float* b_ptr = B.data();
        float* c_ptr = C.data();
        size_t total = A.bytes() / sizeof(float);

        // 2. 决定切几块 (根据你是 pool(4) 还是 pool(8))
        // 简单的办法是硬编码 4，或者给 ThreadPool 加一个 .size() 接口
        int num_threads = 4; // 假设你知道大小
        size_t chunk = (total + num_threads - 1) / num_threads;

        // 3. 计数器同步
        // 注意：atomic 不能被拷贝，必须按引用捕获([&])，或者放在堆上用 shared_ptr 管理
        // 因为 lambda 是按值捕获上下文的 ([=])，如果不小心可能导致问题。
        // 但这里 finished_count 在栈上，且主线程会阻塞等待，所以按引用捕获是安全的。
        std::atomic<int> finished_count{ 0 };

        int tasks_launched = 0;
        for (int i = 0; i < num_threads; ++i) {
            size_t start = i * chunk;
            size_t end = std::min(start + chunk, total);

            if (start >= end) break; // 任务太少，后面的线程没活干

            tasks_launched++;

            // 4. 提交任务
            // 关键修正：Lambda 捕获列表
            // [=] 会按值捕获所有外部变量（包括 pointers, start, end），这是我们要的（因为 i 变了，start 变了）
            // [&finished_count] 特别指定 finished_count 按引用捕获（因为我们要修改它，且它是 atomic 不可拷贝）
            pool.enqueue([=, &finished_count] { 
                for (size_t k = start; k < end; ++k) {
                    c_ptr[k] = a_ptr[k] + b_ptr[k]; // 核心计算：C = A + B
                }

                // 报告完工
                finished_count++;
            });
        }

        // 5. 主线程等待所有分块完成
        // 修正逻辑：只等待实际启动的任务数，而不是 num_threads
        // (比如 total=100, threads=400, chunk=1, 则只启动了 100 个任务)
        while (finished_count < tasks_launched) {
            std::this_thread::yield(); 
        }
    }

} // namespace tt
