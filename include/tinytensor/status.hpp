#pragma once
#include <string_view>
#include <string>

namespace tt {

    enum class Status {
        Ok = 0,
        InvalidArgument,
        OutOfMemory,
        AlignmentError,
        CopyFailed,
        Unknown
    };

    inline std::string_view ToString(Status s) {
        switch (s) {
        case Status::Ok: return "Ok";
        case Status::InvalidArgument: return "InvalidArgument";
        case Status::OutOfMemory: return "OutOfMemory";
        case Status::AlignmentError: return "AlignmentError";
        case Status::CopyFailed: return "CopyFailed";
        default: return "Unknown";
        }
    }

    struct Result {
        Status status{ Status::Ok };
        std::string msg{};
        bool ok() const { return status == Status::Ok; }
        static Result OK() { return {}; }
        static Result Error(Status s, std::string m = {}) { return { s, std::move(m) }; }
    };

} // namespace tt
