#ifndef SEARCH_RESULT_HPP
#define SEARCH_RESULT_HPP

#include <vector>

using namespace std;

struct SearchResult {
    vector<int> path;
    vector<int> explored;
};

#endif