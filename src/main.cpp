#include "uuids/uuidv4.hpp"
#include <iostream>

int main()
{
    uuids::uuid_generator generator;
    for (int i = 0; i < 100; i++)
    {
        auto uuid = generator();
        std::cout << "Generated UUID: " << uuid << std::endl;
    }
    return 0;
}
