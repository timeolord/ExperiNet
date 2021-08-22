#include <vector>
#include "Layer.h"
#pragma once

namespace ExperiNet{
    class AbstractNeuralNetwork{
    public:
        std::vector<AbstractLayer*> layers;
        int inputSize;
        int outputSize;
        virtual void print() = 0;
    };
    class DenseFeedForwardNeuralNetwork : public AbstractNeuralNetwork{
    public:
        int layerAmount;
        DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs, int outputs);
        Vector* evaluate(Vector* input);
        void print() override;
    };

    DenseFeedForwardNeuralNetwork::DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs,
                                                                 int outputs){
        inputSize = inputs;
        outputSize = outputs;
        layerAmount = layerAmount;

        // Adds the input layer
        DenseLayer* inputLayer = new DenseLayer(inputs, layerSize);
        layers.push_back(inputLayer);

        DenseLayer* currentLayer = inputLayer;
        for (int i  = 0; i < layerAmount - 1; i++){
            currentLayer->next = new DenseLayer(layerSize, layerSize);
            currentLayer->next->previous = currentLayer;
            currentLayer = currentLayer->next;
            layers.push_back(currentLayer);
        }

        // Adds the output layer
        currentLayer->next = new DenseLayer(layerSize, outputs);
        currentLayer->next->previous = currentLayer;
        layers.push_back(currentLayer->next);
    }
    
    Vector * DenseFeedForwardNeuralNetwork::evaluate(Vector *input) {
        DenseLayer* inputLayer = dynamic_cast<DenseLayer *>(layers.front());
        inputLayer->output = (*input);

        DenseLayer* currentLayer = inputLayer;
        while (currentLayer->next != nullptr){
            currentLayer->next->getOutput();
            currentLayer = currentLayer->next;
        }
        currentLayer->getOutput();
        return &(currentLayer->output);
    }

    void DenseFeedForwardNeuralNetwork::print() {
        int num = 1;
        std::cout << "=========================\n";
        std::cout << "Input Layer:\n";
        DenseLayer* firstLayer = dynamic_cast<DenseLayer *>(layers.front());
        firstLayer->printOutput();
        std::cout << "=========================\n";

        DenseLayer* currentLayer = firstLayer;
        while (currentLayer->next != nullptr){
            std::cout << "Layer " << num << ":\n";
            num++;
            currentLayer->next->print();
            currentLayer = currentLayer->next;
            std::cout << "=========================\n";
        }
    }
}