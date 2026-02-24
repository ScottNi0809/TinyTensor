#include "tinytensor/tensor.hpp"
#include <numeric>
#include <algorithm>

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

} // namespace tt
