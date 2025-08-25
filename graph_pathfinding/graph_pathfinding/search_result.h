#ifndef SEARCH_RESULT_HPP
#define SEARCH_RESULT_HPP

#include <vector>

struct SearchResult {
    std::vector<int> path;
    std::vector<int> explored;
};

#endif