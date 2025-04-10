///< Example : 4. Custom UUID Implementation with Different PRNG
#include <algorithm>
#include <array>
#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <uuids/uuidv4.hpp>
#include <vector>

class xorshift128plus
{
public:
    using result_type = std::uint64_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT64_MAX; }

    explicit xorshift128plus(result_type seed = default_seed())
    {
        state_[0] = splitmix64(seed);
        state_[1] = splitmix64(state_[0]);
    }

    result_type operator()()
    {
        const result_type s0 = state_[0];
        result_type s1 = state_[1];
        const result_type result = s0 + s1;

        s1 ^= s0;
        state_[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
        state_[1] = rotl(s1, 36);

        return result;
    }

private:
    static result_type default_seed()
    {
        return static_cast<result_type>(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }

    static result_type splitmix64(result_type x)
    {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    static result_type rotl(const result_type x, int k) { return (x << k) | (x >> (64 - k)); }

    std::array<result_type, 2> state_;
};

void benchmark_custom_prng()
{
    using xorshift_uuid = uuids::basic_uuid<xorshift128plus>;
    using xorshift_generator = uuids::basic_uuid_generator<xorshift128plus>;

    uuids::uuid_generator std_generator;
    xorshift_generator custom_generator;

    const int NUM_UUIDS = 1000000;
    std::vector<int> indices(NUM_UUIDS);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<uuids::uuid> std_uuids;
    std_uuids.reserve(NUM_UUIDS);

    auto start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                  [&](int) { std_uuids.push_back(std_generator()); });
    auto std_end = std::chrono::high_resolution_clock::now();

    std::vector<xorshift_uuid> custom_uuids;
    custom_uuids.reserve(NUM_UUIDS);

    std::for_each(std::execution::par_unseq, indices.begin(), indices.end(),
                  [&](int) { custom_uuids.push_back(custom_generator()); });
    auto custom_end = std::chrono::high_resolution_clock::now();

    auto std_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(std_end - start).count();
    auto custom_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(custom_end - std_end).count();

    std::cout << "Standard UUID generation: " << NUM_UUIDS << " UUIDs in " << std_duration << "ms ("
              << static_cast<double>(NUM_UUIDS) / std_duration * 1000 << " UUIDs/second)"
              << std::endl;

    std::cout << "Custom PRNG UUID generation: " << NUM_UUIDS << " UUIDs in " << custom_duration
              << "ms (" << static_cast<double>(NUM_UUIDS) / custom_duration * 1000
              << " UUIDs/second)" << std::endl;

    std::cout << "\nStandard UUID samples:" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << std_uuids[i] << std::endl;
    }

    std::cout << "\nCustom UUID samples:" << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        std::cout << custom_uuids[i] << std::endl;
    }
}

int main()
{
    benchmark_custom_prng();
    return 0;
}
