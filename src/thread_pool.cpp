#include "tinytensor/thread_pool.hpp"
#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace tt
{
	struct ThreadPool::Impl 
	{
		std::vector<std::thread> workers_;
		std::queue<std::function<void()>> jobs_;
		std::mutex mutex_;
		std::condition_variable cv_;
		bool stop_ = false;

		Impl(size_t n)
		{
			if (n == 0) n = std::thread::hardware_concurrency();
			if (n == 0) n = 1;

			for (size_t i = 0; i < n; i++)
			{
				workers_.emplace_back([this]
				{
					for (;;)
					{
						std::function<void()> job;
						{
							std::unique_lock<std::mutex> lock(mutex_);
							cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
							if (stop_ && jobs_.empty()) return;
							job = std::move(jobs_.front());
							jobs_.pop();
						}
						job();
					}
				});
			}
		}

		~Impl()
		{
			{
				std::unique_lock<std::mutex> lock(mutex_);
				stop_ = true;
			}
			cv_.notify_all();
			for (auto& worker : workers_) {
				if (worker.joinable()) worker.join();
			}
		}

		void enqueue(std::function<void()> job)
		{
			{
				std::lock_guard<std::mutex> lock(mutex_);
				jobs_.push(std::move(job));
			}
			cv_.notify_one();
		}
	};

	ThreadPool::ThreadPool(std::size_t n)
		: impl_(new Impl(n)) {}

	ThreadPool::~ThreadPool()
	{
		delete impl_;
	}

	void ThreadPool::enqueue(std::function<void()> job)
	{
		impl_->enqueue(std::move(job));
	}

} // namespace tt