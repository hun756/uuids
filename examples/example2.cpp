///< Example: 2. Lock-Free UUID Generation Pool for High-Throughput Systems
#include <array>
#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <uuids/uuidv4.hpp>
#include <vector>

class UuidPool
{
private:
    static constexpr size_t POOL_SIZE = 1024;
    static constexpr size_t REFILL_THRESHOLD = POOL_SIZE / 4;

    std::array<uuids::uuid, POOL_SIZE> pool_;
    std::atomic<size_t> position_{0};
    std::atomic<bool> running_{true};

    std::mutex mutex_;
    std::condition_variable cv_;
    std::unique_ptr<std::thread> refill_thread_;
    uuids::uuid_generator generator_;

public:
    UuidPool() : generator_()
    {
        refill_pool(0, POOL_SIZE);

        refill_thread_ = std::make_unique<std::thread>(
            [this]
            {
                while (running_.load(std::memory_order_relaxed))
                {
                    {
                        std::unique_lock lock(mutex_);
                        cv_.wait(lock,
                                 [this]
                                 {
                                     return !running_.load(std::memory_order_relaxed) ||
                                            position_.load(std::memory_order_acquire) >=
                                                REFILL_THRESHOLD;
                                 });
                    }

                    if (!running_.load(std::memory_order_relaxed))
                        break;

                    size_t pos = position_.load(std::memory_order_acquire);
                    if (pos >= REFILL_THRESHOLD)
                    {
                        refill_pool(0, pos);
                        position_.store(0, std::memory_order_release);
                    }
                }
            });
    }

    ~UuidPool()
    {
        running_.store(false, std::memory_order_relaxed);
        cv_.notify_all();
        if (refill_thread_ && refill_thread_->joinable())
        {
            refill_thread_->join();
        }
    }

    [[nodiscard]] uuids::uuid get()
    {
        size_t pos = position_.fetch_add(1, std::memory_order_acq_rel);

        if (pos >= POOL_SIZE)
        {
            std::unique_lock lock(mutex_);
            cv_.notify_one();
            return generator_();
        }

        if (pos >= REFILL_THRESHOLD)
        {
            cv_.notify_one();
        }

        return pool_[pos];
    }

private:
    void refill_pool(size_t start, size_t end)
    {
        for (size_t i = start; i < end && i < POOL_SIZE; ++i)
        {
            pool_[i] = generator_();
        }
    }
};

void benchmark_uuid_pool()
{
    UuidPool pool;
    const int NUM_THREADS = 8;
    const int UUIDS_PER_THREAD = 100000;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::future<void>> futures;
    for (int i = 0; i < NUM_THREADS; i++)
    {
        futures.push_back(std::async(std::launch::async,
                                     [&pool]() -> void
                                     {
                                         for (int j = 0; j < UUIDS_PER_THREAD; j++)
                                         {
                                             auto id = pool.get();
                                             if (j == UUIDS_PER_THREAD - 1)
                                             {
                                                 volatile auto str = id.str();
                                             }
                                         }
                                     }));
    }

    for (auto& f : futures)
    {
        f.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Generated " << NUM_THREADS * UUIDS_PER_THREAD << " UUIDs in " << duration
              << "ms using " << NUM_THREADS << " threads" << std::endl;
    std::cout << "Rate: "
              << (1000.0 * NUM_THREADS * UUIDS_PER_THREAD) / static_cast<double>(duration)
              << " UUIDs/second" << std::endl;
}

int main()
{
    benchmark_uuid_pool();
    return 0;
}
