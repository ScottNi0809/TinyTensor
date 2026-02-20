#include "tinytensor/thread_pool.hpp"
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace tt {

    struct ThreadPool::Impl {
        std::vector<std::thread> workers_;
        std::queue<std::function<void()>> q_;
        std::mutex m_;
        std::condition_variable cv_;
        bool stop_ = false;

        Impl(size_t n) {
            if (n == 0) n = std::thread::hardware_concurrency();
            if (n == 0) n = 1; // Fallback if hardware_concurrency returns 0

			// “工人”就是线程, 初始化 n 个工人线程
            for (size_t i = 0; i < n; ++i) {
                workers_.emplace_back([this] {
                    for (;;) {
                        std::function<void()> job;
                        {
                            // 1. 关门上锁（排队，一次只能进一个工人）
                            std::unique_lock<std::mutex> lk(m_);

                            // 2. 等待：如果没有任务 job 且 没说要停机，就在这里睡觉（释放锁，让别人进）
                            // 一旦有人喊“来活了”（notify），或者“停机了”，就醒来重新抢锁
                            cv_.wait(lk, [this] { return stop_ || !q_.empty(); });
                            
                            // 3. 检查是否该下班：如果老板喊停 且 任务都做完了
                            if (stop_ && q_.empty()) return;  // 线程函数结束，工人正式离职
                            
                            // 4. 拿任务：
                            // q_.front() 是从任务堆里拿出来的任务
                            // std::move 是“移动”，把任务从队列里彻底拿出来，塞进工人的公文包(job)里
                            job = std::move(q_.front());

                            q_.pop(); // 把这一也撕掉（队列减一）
                        }  // 花括号结束：带着公文包离开保密室，开锁（lk 自动析构解锁）
                        
                        // 5. 干活！
                        // 注意：这行代码在锁外面执行。
                        // 这里的 job() 才是真正去执行用户塞进来的那个 lambda。
                        job();
                    }
                });
            }
        }

        ~Impl() {
            {
                std::lock_guard<std::mutex> lk(m_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto& w : workers_) {
                if (w.joinable()) w.join();
            }
        }

        void enqueue(std::function<void()> job) {
            {
                std::lock_guard<std::mutex> lk(m_);
                q_.push(std::move(job));
            }
            cv_.notify_one();
        }
    };

    ThreadPool::ThreadPool(std::size_t n) 
        : impl_(new Impl(n)) {}

    ThreadPool::~ThreadPool() {
        delete impl_;
    }

    void ThreadPool::enqueue(std::function<void()> job) {
        impl_->enqueue(std::move(job));
    }

} // namespace tt
