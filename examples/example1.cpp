#include <atomic>
#include <chrono>
#include <format>
#include <iostream>
#include <mutex>
#include <shared_mutex>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <uuids/uuidv4.hpp>

struct UuidEqual
{
    bool operator()(const uuids::uuid& lhs, const uuids::uuid& rhs) const
    {
        return std::equal(lhs.bytes().begin(), lhs.bytes().end(), rhs.bytes().begin());
    }
};

class RequestTracker
{
private:
    struct RequestInfo
    {
        std::chrono::steady_clock::time_point start_time;
        std::string endpoint;
        std::string client_ip;
        int status_code{0};
    };

    using RequestMap =
        std::unordered_map<uuids::uuid, RequestInfo, std::hash<uuids::uuid>, UuidEqual>;

    RequestMap active_requests_;
    mutable std::shared_mutex mutex_;
    uuids::uuid_generator generator_;
    std::atomic<std::uint64_t> active_count_{0};
    std::atomic<std::uint64_t> completed_count_{0};

public:
    [[nodiscard]] uuids::uuid start_request(std::string_view endpoint, std::string_view client_ip)
    {
        auto request_id = generator_();
        auto now = std::chrono::steady_clock::now();

        {
            std::unique_lock lock(mutex_);
            active_requests_.emplace(
                request_id, RequestInfo{now, std::string{endpoint}, std::string{client_ip}});
        }

        active_count_.fetch_add(1, std::memory_order_relaxed);
        return request_id;
    }

    void complete_request(const uuids::uuid& id, int status_code)
    {
        std::unique_lock lock(mutex_);
        if (auto it = active_requests_.find(id); it != active_requests_.end())
        {
            auto& info = it->second;
            info.status_code = status_code;

            auto duration = std::chrono::steady_clock::now() - info.start_time;
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

            std::cout << std::format("Request {} completed: {} {} - {} ({}ms)\n", id.str(),
                                     info.endpoint, info.client_ip, status_code, ms);

            active_requests_.erase(it);
            active_count_.fetch_sub(1, std::memory_order_relaxed);
            completed_count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] auto metrics() const
    {
        return std::make_tuple(active_count_.load(std::memory_order_relaxed),
                               completed_count_.load(std::memory_order_relaxed));
    }
};

void simulate_request_flow()
{
    RequestTracker tracker;

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i)
    {
        threads.emplace_back(
            [&tracker, i]()
            {
                for (int j = 0; j < 25; ++j)
                {
                    auto endpoint = std::format("/api/resource/{}", j % 10);
                    auto client = std::format("192.168.1.{}", 100 + i);

                    auto id = tracker.start_request(endpoint, client);

                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50 + (std::rand() % 100)));

                    int status = (std::rand() % 100) < 95 ? 200 : 500;
                    tracker.complete_request(id, status);
                }
            });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    auto [active, completed] = tracker.metrics();
    std::cout << std::format("Final metrics - Active: {}, Completed: {}\n", active, completed);
}

int main()
{
    simulate_request_flow();
    return 0;
}