
#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include "status.hpp"

namespace tt {

    // 简单的对齐分配器，用 RAII 管理
    class AlignedBuffer {
    public:
        AlignedBuffer() = default;
        AlignedBuffer(std::size_t bytes, std::size_t alignment = 64) {
            allocate(bytes, alignment);
        }
        ~AlignedBuffer() { release(); }

		AlignedBuffer(const AlignedBuffer&) = delete;  // 禁止拷贝构造
		AlignedBuffer& operator=(const AlignedBuffer&) = delete;  // 禁止赋值拷贝

		AlignedBuffer(AlignedBuffer&& other) noexcept  // 允许移动构造
            : ptr_(other.ptr_), bytes_(other.bytes_), alignment_(other.alignment_) {
            other.ptr_ = nullptr; other.bytes_ = 0; other.alignment_ = 0;
        }

		AlignedBuffer& operator=(AlignedBuffer&& other) noexcept {  // 允许移动赋值
            if (this != &other) {
                release();
                ptr_ = other.ptr_; bytes_ = other.bytes_; alignment_ = other.alignment_;
                other.ptr_ = nullptr; other.bytes_ = 0; other.alignment_ = 0;
            }
            return *this;
        }

        tt::Result allocate(std::size_t bytes, std::size_t alignment = 64);
        void release();

        void* data() noexcept { return ptr_; }
        const void* data() const noexcept { return ptr_; }
        std::size_t size() const noexcept { return bytes_; }
        std::size_t alignment() const noexcept { return alignment_; }

    private:
        void* ptr_{ nullptr };
        std::size_t bytes_{ 0 };
        std::size_t alignment_{ 0 };
    };

} // namespace tt
