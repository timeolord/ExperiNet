#include <iostream>
#include "Layer.h"
#include "Network.h"
#include "CostFunctions.h"

using namespace Eigen;
using namespace ExperiNet;

void test();

int main()
{
    test();

    auto network = new DenseFeedForwardNeuralNetwork(2, 2, 1, 1, costFunction);

}

void testFeedForwardNetworkEval(){
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

    network->evaluate(&in);

    if (!((((dynamic_cast<DenseLayer*>(lays->front()))->next)->next)->activations).isApprox(out)){
        std::cout << "Feed Forward Network Eval Test Failed!\n";
    }
}

void testFeedForwardNetworkBackpropagation(){
    float (*costFunction)(float, float) = crossEntropy;

    auto network = new DenseFeedForwardNeuralNetwork(2, 2, 1, 1, costFunction);

    std::vector<AbstractLayer*>* lays = &(network->layers);

    ExperiNet::Matrix weights1 = ExperiNet::Matrix(1, 2);
    weights1 << 0.0258648, 0.678224;
    (((dynamic_cast<DenseLayer*>(lays->front()))->next))->weights = weights1;

    Vector bias1 = Vector(2);
    bias1 << 0.22528, -0.407937;
    (((dynamic_cast<DenseLayer*>(lays->front()))->next))->biases = bias1;

    ExperiNet::Matrix weights2 = ExperiNet::Matrix(2, 1);
    weights2 << 0.275105, 0.0485743;
    ((((dynamic_cast<DenseLayer*>(lays->front()))->next))->next)->weights = weights2;

    Vector bias2 = Vector(1);
    bias2 << -0.012834;
    ((((dynamic_cast<DenseLayer*>(lays->front()))->next))->next)->biases = bias2;

    ExperiNet::Vector input = Vector(1);
    input << 1;
    ExperiNet::Vector output = Vector(1);
    output << 0;

    ExperiNet::Vector err1 = Vector(2);
    err1 << 0.019783, 0.00349301;
    ExperiNet::Vector err2 = Vector(1);
    err2 << 0.0719107;

    network->train(&input, &output);

    if (!((((dynamic_cast<DenseLayer*>(lays->front()))->next)->next)->errors).isApprox(err2)){
        std::cout << "Feed Forward Network Back Propagation Test Failed!\n";
    }
    if (!((((dynamic_cast<DenseLayer*>(lays->front()))->next))->errors).isApprox(err1)){
        std::cout << "Feed Forward Network Back Propagation Test Failed!\n";
    }
}

void testLayerEval(){
    auto* layer = new DenseLayer(2, 3);
    layer->weights << -0.5, 0.6, 0.7,
            -0.8, 0.9, 0.10;
    layer->biases << 0.1, 0.1, 0.1;
    layer->previous = new DenseLayer(2, 2);
    layer->previous->activations << 0.10, 0.15;
    layer->getOutput();
    Vector test = Vector(3);
    test << -0.07, 0.295, 0.185;
    if ((layer->activations).isApprox(test)) {
        return;
    }
    std::cout << "Layer Eval Test Failed!\n";
}

void test() {
    testLayerEval();
    testFeedForwardNetworkEval();
    testFeedForwardNetworkBackpropagation();
}