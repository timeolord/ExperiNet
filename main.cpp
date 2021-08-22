#include <iostream>
#include "Layer.h"
#include "Network.h"

using namespace Eigen;
using namespace ExperiNet;

void test();

int main()
{
    test();


}

bool testFeedForwardNetworkEval(){
    auto network = new DenseFeedForwardNeuralNetwork(2, 2, 2, 1);
    std::vector<AbstractLayer*>* lays = &(network->layers);

    ExperiNet::Matrix weights1 = ExperiNet::Matrix(2, 2);
    weights1 << -0.514226, 0.608353,
              -0.725537, -0.686642;
    ((dynamic_cast<DenseLayer*>(lays->front()))->next)->weights = weights1;

    Vector bias1 = Vector(2);
    bias1 << -0.198111, -0.740419;
    ((dynamic_cast<DenseLayer*>(lays->front()))->next)->biases = bias1;

    ExperiNet::Matrix weights2 = ExperiNet::Matrix(2, 1);
    weights2 << -0.782382, 0.997849;
    (((dynamic_cast<DenseLayer*>(lays->front()))->next)->next)->weights = weights2;

    Vector bias2 = Vector(1);
    bias2 << -0.563486;
    (((dynamic_cast<DenseLayer*>(lays->front()))->next)->next)->biases = bias2;

    Vector in = Vector(2);
    in << 1, 2;

    Vector out = Vector(1);
    out << -0.372984;

    Vector * output = network->evaluate(&in);

    if (!((((dynamic_cast<DenseLayer*>(lays->front()))->next)->next)->output).isApprox(out)){
        std::cout << "Feed Forward Network Eval Test Failed!\n";
    }
}

void testLayerEval(){
    auto* layer = new DenseLayer(2, 3);
    layer->weights << -0.5, 0.6, 0.7,
            -0.8, 0.9, 0.10;
    layer->biases << 0.1, 0.1, 0.1;
    layer->previous = new DenseLayer(2, 2);
    layer->previous->output << 0.10, 0.15;
    layer->getOutput();
    Vector test = Vector(3);
    test << -0.07, 0.295, 0.185;
    if ((layer->output).isApprox(test)) {
        return;
    }
    std::cout << "Layer Eval Test Failed!\n";
}

void test() {
    testLayerEval();
    testFeedForwardNetworkEval();
}