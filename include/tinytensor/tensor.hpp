
#pragma once
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include "status.hpp"
#include "aligned_alloc.hpp"
#include "thread_pool.hpp"

namespace tt {

    enum class Layout { AoS, SoA };

    struct Shape {
        std::vector<std::size_t> dims; // e.g., {N, C}
    };

    class Tensor {
    public:
        Tensor() = default;
        ~Tensor() = default;

        // 禁止拷贝，允许移动（像 GPU buffer）
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;
        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

        static Result CreateFloat32(Shape shape, Layout layout, Tensor& out);

        Result copy_from(const void* src, std::size_t bytes, std::size_t offset = 0);
        Result copy_to(void* dst, std::size_t bytes, std::size_t offset = 0) const;

        float* data() { return reinterpret_cast<float*>(buf_.data()); }
        const float* data() const { return reinterpret_cast<const float*>(buf_.data()); }

        std::size_t bytes() const { return buf_.size(); }
        const Shape& shape() const { return shape_; }
        Layout layout() const { return layout_; }
        std::size_t stride(std::size_t dim) const { return strides_[dim]; }

        static void ParallelAdd(tt::ThreadPool& pool, const tt::Tensor& A, const tt::Tensor& B, tt::Tensor& C);

    private:
        AlignedBuffer buf_{};
        Shape shape_{};
        std::vector<std::size_t> strides_{};
        Layout layout_{ Layout::AoS };
    };

} // namespace tt
