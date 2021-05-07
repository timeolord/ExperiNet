#include <iostream>
#include "Layer.h"

using namespace Eigen;
using namespace ExperiNet;

int main()
{
    auto* layer = new DenseLayer(2, 3);
    layer->weights << -0.5, 0.6, 0.7,
                      -0.8, 0.9, 0.10;
    std::cout << layer->weights << std::endl;
    layer->biases << 0.1, 0.1, 0.1;
    std::cout << layer->biases << std::endl;
    layer->previous = new DenseLayer(2, 2);
    layer->previous->output << 0.10, 0.15;
    std::cout << layer->previous->output << std::endl;
    layer->getOutput();
}

void test(){

}