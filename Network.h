#include <vector>
#include "Layer.h"
#include "CostFunctions.h"
#include <iterator>
#include <algorithm>
#include <random>
#pragma once

namespace ExperiNet{
    class AbstractNeuralNetwork{
    public:
        std::vector<AbstractLayer*> layers;
        int inputSize;
        int outputSize;
        virtual void printMatrices() = 0;
        virtual void printStructure() = 0;
    };
    class feedForwardNeuralNetwork : public AbstractNeuralNetwork{
    public:
        int layerAmount;
        CostFunctions::CostFunction* cost;
        int minibatchSize;
        float learningRate;

        feedForwardNeuralNetwork(CostFunctions::CostFunction* cost, float learningRate, int minibatchSize);
        feedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs, int outputs, float learningRate, int minibatchSize, CostFunctions::CostFunction* cost);
        Vector* evaluate(Vector* input);
        void printMatrices() override;
        void printStructure() override;
        void train(Vector *input, Vector *output, int epochs);
        void train(std::vector<Vector>* input, std::vector<Vector>* output, int epochs, bool debug);
        void backPropagate(Vector *output);
        void stochasticGradientDescent();
        void add(DenseLayer* layer);
        void printOutput();
        float calculateLoss(std::vector<Vector>* input, std::vector<Vector>* output, int epochs);
    };

    feedForwardNeuralNetwork::feedForwardNeuralNetwork(CostFunctions::CostFunction* cost, float learningRate, int minibatchSize){
        this->cost = cost;
        this->learningRate = learningRate;
        this->minibatchSize = minibatchSize;
    }

    feedForwardNeuralNetwork::feedForwardNeuralNetwork(int layerAmount, int layerSize, int inputs,
                                                       int outputs, float learningRate, int minibatchSize, CostFunctions::CostFunction* cost){
        this->inputSize = inputs;
        this->outputSize = outputs;
        this->layerAmount = layerAmount;
        this->learningRate = learningRate;
        this->minibatchSize = minibatchSize;

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

        //Cost function
        this->cost = cost;
    }

    
    Vector * feedForwardNeuralNetwork::evaluate(Vector *input) {
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

    void feedForwardNeuralNetwork::printMatrices() {
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
            currentLayer->next->printMatrix();
            currentLayer = currentLayer->next;
            std::cout << "=========================\n";
        }
    }

    void feedForwardNeuralNetwork::printStructure() {
        int num = 1;
        std::cout << "=========================\n";
        std::cout << "Input Size: ";
        DenseLayer* firstLayer = dynamic_cast<DenseLayer *>(layers.front());
        std::cout << firstLayer->neurons;
        std::cout << "\n=========================\n";

        DenseLayer* currentLayer = firstLayer;
        while (currentLayer->next != nullptr){
            std::cout << "Layer " << num << ":\n";
            num++;
            currentLayer->next->printStructure();
            currentLayer = currentLayer->next;
            std::cout << "=========================\n";
        }
    }

    void feedForwardNeuralNetwork::backPropagate(Vector *output) {

        DenseLayer* currentLayer = dynamic_cast<DenseLayer *>(this->layers.back());

        Vector* errors = &(currentLayer->errors);

        //Takes care of the last case
        for (int i = 0; i < outputSize; i++){
            //Computes the cost prime for each neuron
            (*errors)(i) = cost->derivative(currentLayer->activations(i), (*output)(i));
            //Multiples by the derivative of the activation function
            (*errors)(i) *= currentLayer->activationFunction->derivative(currentLayer->weightedInputs(i));
        }
        //Collects and calculates the average gradient
        Matrix gradientAverage = Matrix::Zero(currentLayer->weightGradients.rows(), currentLayer->weightGradients.cols());

        //Computes the gradients for each neuron
        for (int i = 0; i < currentLayer->weightGradients.cols(); i++) {
            for (int j = 0; j < currentLayer->weightGradients.rows(); j++) {
                gradientAverage(j, i) = currentLayer->previous->activations(i) * currentLayer->errors(j);
            }
        }
        currentLayer->gradients.push_back(gradientAverage);

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

            //Computes the gradients for each neuron
            gradientAverage = Matrix::Zero(currentLayer->weightGradients.rows(), currentLayer->weightGradients.cols());

            for (int i = 0; i < currentLayer->weightGradients.cols(); i++) {
                for (int j = 0; j < currentLayer->weightGradients.rows(); j++) {
                    gradientAverage(j, i) = currentLayer->previous->activations(i) * currentLayer->errors(j);
                }
            }
            currentLayer->gradients.push_back(gradientAverage);

            currentLayer = currentLayer->previous;
        }


    }

    void feedForwardNeuralNetwork::train(Vector *input, Vector *output, int epochs) {
        for (int i = 0; i < epochs; i++){
            this->evaluate(input);
            this->backPropagate(output);
            this->stochasticGradientDescent();
        }
    }

    void feedForwardNeuralNetwork::stochasticGradientDescent() {
        DenseLayer* inputLayer = dynamic_cast<DenseLayer *>(layers.front());
        DenseLayer* currentLayer = inputLayer;

        while (currentLayer->next != nullptr){
            //Calculates the average gradient
            for (int i = 0; i < currentLayer->weightGradients.rows(); i++){
                for (int j = 0; j < currentLayer->weightGradients.cols(); j++){
                    float average = 0;
                    for (int k = 0; k < currentLayer->gradients.size(); k++){
                        average += (currentLayer->gradients.at(k))(i, j);
                    }
                    average = average / minibatchSize;
                    currentLayer->weightGradients(i, j) = average;
                }
            }
            currentLayer->gradients.clear();

            currentLayer->next->gradientDescent(learningRate);
            currentLayer = currentLayer->next;
        }
        //Calculates the average gradient
        for (int i = 0; i < currentLayer->weightGradients.rows(); i++){
            for (int j = 0; j < currentLayer->weightGradients.cols(); j++){
                float average = 0;
                for (int k = 0; k < currentLayer->gradients.size(); k++){
                    average += (currentLayer->gradients.at(k))(i, j);
                }
                average = average / minibatchSize;
                currentLayer->weightGradients(i, j) = average;
            }
        }
        currentLayer->gradients.clear();

        currentLayer->gradientDescent(learningRate);
    }

    void feedForwardNeuralNetwork::add(DenseLayer* layer) {
        //First layer added
        if (this->layers.empty()){
            this->inputSize = layer->neurons;
            this->layers.push_back(layer);
            this->layerAmount = 0;
        }
        else {
            DenseLayer* lastLayer = dynamic_cast<DenseLayer *>(this->layers.back());
            lastLayer->next = layer;
            layer->previous = lastLayer;
            this->layers.push_back(layer);
            this->layerAmount++;
            this->outputSize = layer->neurons;
        }
    }

    void feedForwardNeuralNetwork::train(std::vector<Vector>* input, std::vector<Vector>* output, int epochs, bool debug) {

        std::random_device randomDevice; // obtain a random number from hardware
        std::mt19937 generator(randomDevice()); // seed the generator
        std::uniform_int_distribution<> distribution(0, input->size() - 1);

        for (int j = 0; j < epochs; j++){
            if (debug && j % (epochs / 10) == 1){
                float temp = calculateLoss(input, output, 10);
                std::cout << "Loss after " << j << " epochs: " << temp << "\n";
            }
            for(int i = 0; i < minibatchSize; i++){
                int randomNumber = distribution(generator);
                this->evaluate(&input->at(randomNumber));
                this->backPropagate(&output->at(randomNumber));
            }
            this->stochasticGradientDescent();
        }
        if (debug){
            float temp = calculateLoss(input, output, 10);
            std::cout << "Final Loss: " << temp << "\n";
        }
    }

    void feedForwardNeuralNetwork::printOutput() {
        std::cout << "Input:\n";
        dynamic_cast<DenseLayer*>(this->layers.front())->printOutput();
        std::cout << "Output:\n";
        dynamic_cast<DenseLayer*>(this->layers.back())->printOutput();
    }

    float feedForwardNeuralNetwork::calculateLoss(std::vector<Vector>* input, std::vector<Vector>* output, int epochs) {

        std::random_device randomDevice;
        std::mt19937 generator(randomDevice());
        std::uniform_int_distribution<> distribution(0, input->size() - 1);

        float outsideAverage = 0;
        for(int i = 0; i < epochs; i++){
            float insideAverage = 0;
            int randomNumber = distribution(generator);
            Vector tempOutput = *(this->evaluate(&input->at(randomNumber)));
            for (int j = 0; j < tempOutput.size(); j++){
                insideAverage += this->cost->cost(tempOutput(j), output->at(randomNumber)(j));
            }
            insideAverage = insideAverage / tempOutput.size();
            outsideAverage += insideAverage;
        }
        return outsideAverage / epochs;
    }
}