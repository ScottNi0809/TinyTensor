
#include "tinytensor/aligned_alloc.hpp"
#include <cstdlib>
#include <cstring>

namespace tt {

    static bool is_power_of_two(std::size_t x) {
        return x && ((x & (x - 1)) == 0);
    }

    Result AlignedBuffer::allocate(std::size_t bytes, std::size_t alignment) {
        if (bytes == 0) return Result::Error(Status::InvalidArgument, "bytes=0");
        if (!is_power_of_two(alignment)) return Result::Error(Status::AlignmentError, "alignment not power of two");

#if defined(_MSC_VER)
        void* p = _aligned_malloc(bytes, alignment);
        if (!p) return Result::Error(Status::OutOfMemory, "aligned malloc failed");
        ptr_ = p;
#else
        void* p = nullptr;
        if (posix_memalign(&p, alignment, bytes) != 0) {
            return Result::Error(Status::OutOfMemory, "posix_memalign failed");
        }
        ptr_ = p;
#endif
        bytes_ = bytes; 
        alignment_ = alignment;
        return Result::OK();
    }

    void AlignedBuffer::release() {
        if (!ptr_) return;
#if defined(_MSC_VER)
        _aligned_free(ptr_);
#else
        std::free(ptr_);
#endif
        ptr_ = nullptr; 
        bytes_ = 0; 
        alignment_ = 0;
    }

} // namespace tt
