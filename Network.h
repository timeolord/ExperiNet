#include <vector>
#include "Layer.h"
#include "CostFunctions.h"
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
        float (*cost)(float, float);
        int minibatchSize;
        int learningRate;

        DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs, int outputs);
        DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs, int outputs, float (*cost)(float, float));
        Vector* evaluate(Vector* input);
        void print() override;
        void train(Vector *input, Vector *output);
        void backPropagate(Vector *output);
        void stochasticGradientDescent();
    };

    DenseFeedForwardNeuralNetwork::DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs,
                                                                 int outputs){
        this->inputSize = inputs;
        this->outputSize = outputs;
        this->layerAmount = layerAmount;

        // Adds the input layer
        DenseLayer* inputLayer = new DenseLayer(1, inputs);
        layers.push_back(inputLayer);

        // Adds the first layer
        inputLayer->next = new DenseLayer(inputs, layerSize);
        inputLayer->next->previous = inputLayer;
        layers.push_back(inputLayer->next);

        DenseLayer* currentLayer = inputLayer->next;
        for (int i  = 0; i < layerAmount - 2; i++){
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

    DenseFeedForwardNeuralNetwork::DenseFeedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs,
                                                                 int outputs, float (*cost)(float, float)) :
                                                                 DenseFeedForwardNeuralNetwork(layerAmount, layerSize,
                                                                                               inputs, outputs) {
        this->cost = cost;
    }

    
    Vector * DenseFeedForwardNeuralNetwork::evaluate(Vector *input) {
        DenseLayer* inputLayer = dynamic_cast<DenseLayer *>(layers.front());
        inputLayer->activations = (*input);

        DenseLayer* currentLayer = inputLayer;
        while (currentLayer->next != nullptr){
            currentLayer->next->getOutput();
            currentLayer = currentLayer->next;
        }
        currentLayer->getOutput();
        return &(currentLayer->activations);
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

    void DenseFeedForwardNeuralNetwork::backPropagate(Vector *output) {

        DenseLayer* currentLayer = dynamic_cast<DenseLayer *>(this->layers.back());

        Vector* errors = &(currentLayer->errors);

        //Takes care of the last case
        for (int i = 0; i < outputSize; i++){
            //Computes the cost for each neuron
            (*errors)(i) = cost(currentLayer->activations(i), (*output)(i));
            //Multiples by the derivative of the activation function
            (*errors)(i) *= currentLayer->activationFunction->derivative((*errors)(i));
        }
        currentLayer = currentLayer->previous;

        while(currentLayer->previous != nullptr){
            for (int i = 0; i < layerAmount; i++){
                //Computes activation function prime vector
                Vector primes = Vector(currentLayer->neurons);
                for (int j = 0; j < currentLayer->neurons; j++){
                    primes(j) =  currentLayer->activationFunction->derivative(currentLayer->weightedInputs(j));
                }
                //Multiplies the weights with the error vector (from the previous layer) and hadamard product with the activation function prime vector
                currentLayer->errors = (currentLayer->next->weights * currentLayer->next->errors).cwiseProduct(primes);
            }
            currentLayer = currentLayer->previous;
        }

    }

    void DenseFeedForwardNeuralNetwork::train(Vector *input, Vector *output) {
        this->evaluate(input);
        this->backPropagate(output);
        this->stochasticGradientDescent();
    }

    void DenseFeedForwardNeuralNetwork::stochasticGradientDescent() {

    }
}