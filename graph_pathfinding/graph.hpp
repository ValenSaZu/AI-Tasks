#include <vector>

using namespace std;

struct edge{
    node* destination;
    int weight;
    int distance;
}
struct node{
    int id;
    vector<edge> neighbors;
}

class graph{
    vector<node> nodes;
}