#ifndef RNG_UTILS_HPP
#define RNG_UTILS_HPP

#include <random>

class RNG {
public:
    RNG() : rng(std::random_device{}()) {}

    // Generate a random float in the range [min, max)
    float randomFloat(float min, float max) {
        std::uniform_real_distribution<float> dist(min, max);
        return dist(rng);
    }

    // Generate a random integer in the range [min, max)
    int randomInt(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max - 1);
        return dist(rng);
    }

private:
    std::mt19937 rng; // Mersenne Twister random number generator
};

#endif