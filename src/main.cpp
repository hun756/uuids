#include <cstdio>
#include <iostream>
#include "uuids/uuidv4.hpp"

int main()
{
    // (void)fprintf(stdout, "Hello from the uuids library!\n");
    uuids::uuid_generator generator;
    for (int i = 0; i < 100; i++)
    {
        auto uuid = generator();
        std::cout << "Generated UUID: " << uuid << std::endl;
    }
    return 0;
}
