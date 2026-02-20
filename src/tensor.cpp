
#include "tinytensor/tensor.hpp"
#include <numeric>
#include <algorithm>

namespace tt {

    static std::size_t product(const std::vector<std::size_t>& v) {
        return std::accumulate(v.begin(), v.end(), std::size_t{ 1 }, std::multiplies<>{});
    }

    // 根据你要求的形状（几乘几），计算出需要多少内存、怎么排布数据，并真正向系统申请这块内存。
    Result Tensor::CreateFloat32(Shape shape, Layout layout, Tensor& out) {
        if (shape.dims.empty()) 
            return Result::Error(Status::InvalidArgument, "dims empty");

        out.shape_ = shape;
        out.layout_ = layout;

        // 计算字节数与步幅；演示 AoS 与 SoA 的差异
        // 例：shape {N, C}，每元素 4字节
        std::size_t N = shape.dims[0];
        std::size_t total_elems = product(shape.dims);

        // 简化：只处理 1D/2D
        out.strides_.resize(shape.dims.size());

		std::size_t bytes = total_elems * sizeof(float);  // 计算总字节数

        auto res = out.buf_.allocate(bytes, /*alignment=*/64);
        if (!res.ok()) 
            return res;

        // 初始化数据为 0
        std::memset(out.buf_.data(), 0, out.buf_.size());

        if (shape.dims.size() == 1) {
            out.strides_[0] = sizeof(float);
        }
        else if (shape.dims.size() == 2) {
			std::size_t C = shape.dims[1];  // 每行元素数（通道数）
            if (layout == Layout::AoS) {
                // [N][C] 连续，每行步幅 C*sizeof(float)
                out.strides_[1] = sizeof(float);          // 每个通道相邻
                out.strides_[0] = C * sizeof(float);      // 行步幅
            }
            else {
                // SoA：每个通道都连续，需要更复杂的索引方式
                // 简化：我们仍给出理论步幅，真实访问需转换索引
                out.strides_[1] = N * sizeof(float);      // 跨样本
                out.strides_[0] = sizeof(float);          // 连续样本
            }
        }
        else {
            return Result::Error(Status::InvalidArgument, "only 1D/2D supported in demo");
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
