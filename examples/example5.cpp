#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <uuids/uuidv4.hpp>
#include <vector>

struct UuidEqual
{
    bool operator()(const uuids::uuid& lhs, const uuids::uuid& rhs) const
    {
        const auto& bytes1 = lhs.bytes();
        const auto& bytes2 = rhs.bytes();

        for (size_t i = 0; i < 16; ++i)
        {
            if (bytes1[i] != bytes2[i])
                return false;
        }
        return true;
    }
};

class ShardedDatabase
{
public:
    static constexpr size_t SHARD_COUNT = 16;
    using ShardId = uint8_t;

private:
    std::array<std::unordered_map<uuids::uuid, std::string, std::hash<uuids::uuid>, UuidEqual>,
               SHARD_COUNT>
        shards_;

    mutable std::array<std::mutex, SHARD_COUNT> shard_mutexes_;

    uuids::uuid_generator generator_;
    std::mutex generator_mutex_;

    ShardId get_shard_id(const uuids::uuid& id) const
    {
        const auto& bytes = id.bytes();
        uint8_t hash = bytes[0] ^ bytes[3] ^ bytes[8] ^ bytes[15];
        return hash & (SHARD_COUNT - 1);
    }

public:
    uuids::uuid insert(const std::string& value)
    {
        uuids::uuid id;
        {
            std::lock_guard<std::mutex> lock(generator_mutex_);
            id = generator_();
        }

        auto shard = get_shard_id(id);

        {
            std::lock_guard<std::mutex> lock(shard_mutexes_[shard]);
            shards_[shard].emplace(id, value);
        }

        return id;
    }

    bool insert(const uuids::uuid& id, const std::string& value)
    {
        auto shard = get_shard_id(id);

        std::lock_guard<std::mutex> lock(shard_mutexes_[shard]);
        auto [_, inserted] = shards_[shard].emplace(id, value);
        return inserted;
    }

    std::optional<std::string> get(const uuids::uuid& id) const
    {
        auto shard = get_shard_id(id);

        std::lock_guard<std::mutex> lock(shard_mutexes_[shard]);
        auto it = shards_[shard].find(id);
        if (it == shards_[shard].end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    bool update(const uuids::uuid& id, const std::string& value)
    {
        auto shard = get_shard_id(id);

        std::lock_guard<std::mutex> lock(shard_mutexes_[shard]);
        auto it = shards_[shard].find(id);
        if (it == shards_[shard].end())
        {
            return false;
        }
        it->second = value;
        return true;
    }

    bool remove(const uuids::uuid& id)
    {
        auto shard = get_shard_id(id);

        std::lock_guard<std::mutex> lock(shard_mutexes_[shard]);
        return shards_[shard].erase(id) > 0;
    }

    std::vector<size_t> get_shard_sizes() const
    {
        std::vector<size_t> sizes;
        sizes.reserve(SHARD_COUNT);

        for (size_t i = 0; i < SHARD_COUNT; ++i)
        {
            std::lock_guard<std::mutex> lock(shard_mutexes_[i]);
            sizes.push_back(shards_[i].size());
        }

        return sizes;
    }

    size_t total_size() const
    {
        size_t total = 0;
        for (size_t i = 0; i < SHARD_COUNT; ++i)
        {
            std::lock_guard<std::mutex> lock(shard_mutexes_[i]);
            total += shards_[i].size();
        }
        return total;
    }

    void analyze_distribution() const
    {
        auto sizes = get_shard_sizes();
        size_t total = 0;

        for (const auto& size : sizes)
        {
            total += size;
        }

        std::cout << "UUID Distribution Analysis:" << std::endl;
        std::cout << "Total items: " << total << std::endl;

        if (total == 0)
            return;

        double expected = static_cast<double>(total) / SHARD_COUNT;
        double variance = 0.0;

        std::cout << "Shard sizes:" << std::endl;
        for (size_t i = 0; i < sizes.size(); ++i)
        {
            double pct = static_cast<double>(sizes[i]) / static_cast<double>(total) * 100.0;
            std::cout << "  Shard " << i << ": " << sizes[i] << " items (" << std::fixed
                      << std::setprecision(2) << pct << "%)" << std::endl;

            double diff = static_cast<double>(sizes[i]) - expected;
            variance += diff * diff;
        }

        variance /= SHARD_COUNT;
        double stddev = std::sqrt(variance);
        double cv = (expected > 0) ? (stddev / expected * 100.0) : 0.0;

        std::cout << "Statistics:" << std::endl;
        std::cout << "  Expected items per shard: " << expected << std::endl;
        std::cout << "  Standard deviation: " << stddev << " items" << std::endl;
        std::cout << "  Coefficient of variation: " << cv << "%" << std::endl;

        if (cv < 5.0)
        {
            std::cout << "Distribution quality: Excellent" << std::endl;
        }
        else if (cv < 10.0)
        {
            std::cout << "Distribution quality: Good" << std::endl;
        }
        else if (cv < 20.0)
        {
            std::cout << "Distribution quality: Acceptable" << std::endl;
        }
        else
        {
            std::cout << "Distribution quality: Poor - consider reviewing the sharding algorithm"
                      << std::endl;
        }
    }
};

bool simulate_database_load()
{
    ShardedDatabase db;
    const int NUM_ITEMS = 10000;
    const int NUM_THREADS = 4;
    const int TIMEOUT_SECONDS = 20;

    std::vector<std::future<bool>> futures;
    futures.reserve(NUM_THREADS);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::mt19937> thread_gens;

    for (int i = 0; i < NUM_THREADS; ++i)
    {
        std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd()};
        thread_gens.emplace_back(seed);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t t = 0; t < NUM_THREADS; ++t)
    {
        futures.push_back(std::async(
            std::launch::async,
            [&db, t, &thread_gens]() -> bool
            {
                try
                {
                    auto& thread_gen = thread_gens[t];
                    std::uniform_int_distribution<size_t> dist;

                    const size_t items_per_thread = NUM_ITEMS / NUM_THREADS;
                    const size_t start_idx = t * items_per_thread;
                    const size_t end_idx =
                        (t == NUM_THREADS - 1) ? NUM_ITEMS : start_idx + items_per_thread;

                    std::vector<uuids::uuid> ids;
                    ids.reserve(end_idx - start_idx);

                    for (size_t i = start_idx; i < end_idx; ++i)
                    {
                        auto value = "Item-" + std::to_string(i);
                        ids.push_back(db.insert(value));
                    }

                    int updates = std::min(50, static_cast<int>(ids.size() / 20)); // 5% of items
                    for (int i = 0; i < updates && !ids.empty(); ++i)
                    {
                        size_t idx = dist(thread_gen) % ids.size();
                        auto value =
                            "Updated-" + std::to_string(start_idx + static_cast<size_t>(i));
                        db.update(ids[idx], value);
                    }

                    int removals = std::min(25, static_cast<int>(ids.size() / 40)); // 2.5% of items
                    for (int i = 0; i < removals && !ids.empty(); ++i)
                    {
                        size_t idx = dist(thread_gen) % ids.size();
                        db.remove(ids[idx]);
                        std::swap(ids[idx], ids.back());
                        ids.pop_back();
                    }

                    return true;
                }
                catch (const std::exception& e)
                {
                    std::cerr << "Thread " << t << " error: " << e.what() << std::endl;
                    return false;
                }
                catch (...)
                {
                    std::cerr << "Thread " << t << " unknown error" << std::endl;
                    return false;
                }
            }));
    }

    bool all_successful = true;
    for (size_t i = 0; i < futures.size(); ++i)
    {
        auto status = futures[i].wait_for(std::chrono::seconds(TIMEOUT_SECONDS));
        if (status != std::future_status::ready)
        {
            std::cerr << "Thread " << i << " could not complete within timeout period!"
                      << std::endl;
            all_successful = false;
        }
        else
        {
            try
            {
                bool thread_success = futures[i].get();
                if (!thread_success)
                {
                    all_successful = false;
                }
            }
            catch (const std::exception& e)
            {
                std::cerr << "Thread " << i << " exception: " << e.what() << std::endl;
                all_successful = false;
            }
            catch (...)
            {
                std::cerr << "Thread " << i << " unknown exception" << std::endl;
                all_successful = false;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Processed " << NUM_ITEMS << " operations in " << duration << "ms" << std::endl;
    std::cout << "Rate: " << static_cast<double>(NUM_ITEMS) / static_cast<double>(duration) * 1000
              << " ops/second" << std::endl;

    db.analyze_distribution();

    return all_successful;
}

int main()
{
    try
    {
        bool success = simulate_database_load();
        if (success)
        {
            std::cout << "Simulation completed successfully." << std::endl;
        }
        else
        {
            std::cout << "Simulation completed with errors." << std::endl;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "An unknown error occurred!" << std::endl;
        return 1;
    }

    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.get();

    return 0;
}
