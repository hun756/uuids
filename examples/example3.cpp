///< Example: 3. Time-Based Key-Value Cache with UUIDs
#include <chrono>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <uuids/uuidv4.hpp>

struct UuidEqual
{
    bool operator()(const uuids::uuid& lhs, const uuids::uuid& rhs) const
    {
        return std::equal(lhs.bytes().begin(), lhs.bytes().end(), rhs.bytes().begin());
    }
};

template <typename ValueType>
class UuidCache
{
public:
    using uuid_type = uuids::uuid;
    using clock_type = std::chrono::steady_clock;
    using time_point = clock_type::time_point;
    using duration = std::chrono::milliseconds;

    struct CacheEntry
    {
        ValueType value;
        time_point expiry;
        std::function<void(const uuid_type&, const ValueType&)> on_expire;
    };

private:
    struct EntryRecord
    {
        time_point expiry;
        typename std::list<uuid_type>::iterator lru_it;
    };

    std::unordered_map<uuid_type, CacheEntry, std::hash<uuid_type>, UuidEqual> entries_;

    std::list<uuid_type> lru_list_;
    std::unordered_map<uuid_type, EntryRecord, std::hash<uuid_type>, UuidEqual> records_;

    mutable std::shared_mutex mutex_;

    const size_t max_size_;
    const duration default_ttl_;

    uuids::uuid_generator generator_;

public:
    explicit UuidCache(size_t max_size = 1000, duration default_ttl = std::chrono::minutes(10))
        : max_size_(max_size), default_ttl_(default_ttl)
    {
    }

    [[nodiscard]] uuid_type
    insert(const ValueType& value, std::optional<duration> ttl = std::nullopt,
           std::function<void(const uuid_type&, const ValueType&)> on_expire = nullptr)
    {
        auto uuid = generator_();
        auto actual_ttl = ttl.value_or(default_ttl_);
        auto expiry = clock_type::now() + actual_ttl;

        std::unique_lock lock(mutex_);

        if (entries_.size() >= max_size_)
        {
            evict_oldest();
        }

        entries_.emplace(uuid, CacheEntry{value, expiry, on_expire});
        auto lru_it = lru_list_.insert(lru_list_.begin(), uuid);
        records_.emplace(uuid, EntryRecord{expiry, lru_it});

        return uuid;
    }

    [[nodiscard]] std::optional<ValueType> get(const uuid_type& uuid)
    {
        std::unique_lock lock(mutex_);

        auto it = entries_.find(uuid);
        if (it == entries_.end())
        {
            return std::nullopt;
        }

        if (clock_type::now() > it->second.expiry)
        {
            remove_locked(uuid);
            return std::nullopt;
        }

        lru_list_.erase(records_[uuid].lru_it);
        auto lru_it = lru_list_.insert(lru_list_.begin(), uuid);
        records_[uuid].lru_it = lru_it;

        return it->second.value;
    }

    bool remove(const uuid_type& uuid)
    {
        std::unique_lock lock(mutex_);
        return remove_locked(uuid);
    }

    size_t cleanup_expired()
    {
        std::unique_lock lock(mutex_);
        size_t removed = 0;
        auto now = clock_type::now();

        auto it = lru_list_.rbegin();
        while (it != lru_list_.rend())
        {
            const auto& uuid = *it;
            auto record_it = records_.find(uuid);

            if (record_it != records_.end() && record_it->second.expiry < now)
            {
                auto temp = it;
                ++it;

                if (remove_locked(uuid))
                {
                    removed++;
                }
            }
            else
            {
                ++it;
            }
        }

        return removed;
    }

    [[nodiscard]] size_t size() const
    {
        std::shared_lock lock(mutex_);
        return entries_.size();
    }

    [[nodiscard]] size_t capacity() const { return max_size_; }

private:
    bool remove_locked(const uuid_type& uuid)
    {
        auto entry_it = entries_.find(uuid);
        if (entry_it == entries_.end())
        {
            return false;
        }

        if (entry_it->second.on_expire)
        {
            try
            {
                entry_it->second.on_expire(uuid, entry_it->second.value);
            }
            catch (...)
            {
                // Ignore callback exceptions
            }
        }

        auto record_it = records_.find(uuid);
        if (record_it != records_.end())
        {
            lru_list_.erase(record_it->second.lru_it);
            records_.erase(record_it);
        }

        entries_.erase(entry_it);
        return true;
    }

    void evict_oldest()
    {
        if (lru_list_.empty())
            return;

        auto oldest = lru_list_.back();
        remove_locked(oldest);
    }
};

void demonstrate_uuid_cache()
{
    UuidCache<std::shared_ptr<std::string>> cache(500, std::chrono::seconds(300));

    auto id1 = cache.insert(
        std::make_shared<std::string>("Short-lived value"),
        std::chrono::milliseconds(100), // 100ms TTL
        [](const uuids::uuid& id, const std::shared_ptr<std::string>& value)
        { std::cout << "Expired: " << *value << " (UUID: " << id << ")" << std::endl; });

    auto id2 = cache.insert(std::make_shared<std::string>("Standard value"));

    auto id3 = cache.insert(std::make_shared<std::string>("Long-lived value"),
                            std::chrono::minutes(60) // 1 hour TTL
    );

    std::cout << "Initial access:" << std::endl;
    if (auto value = cache.get(id1))
    {
        std::cout << "Value 1: " << **value << std::endl;
    }

    if (auto value = cache.get(id2))
    {
        std::cout << "Value 2: " << **value << std::endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    std::cout << "\nAfter waiting:" << std::endl;
    if (auto value = cache.get(id1))
    {
        std::cout << "Value 1: " << **value << std::endl;
    }
    else
    {
        std::cout << "Value 1: expired" << std::endl;
    }

    size_t cleaned = cache.cleanup_expired();
    std::cout << "Cleaned up " << cleaned << " expired entries" << std::endl;

    std::cout << "Cache size: " << cache.size() << "/" << cache.capacity() << std::endl;
}

int main()
{
    demonstrate_uuid_cache();
    return 0;
}
