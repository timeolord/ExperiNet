#pragma once
#include <eigen-3.3.9/Eigen/Dense>
#include "ActivationFunctions.h"
#include <cstdlib>

namespace ExperiNet {
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;

    class AbstractLayer {
        virtual void getOutput() = 0;
        virtual void print() = 0;
    };

    class DenseLayer : public AbstractLayer {
    public:
        int neurons;

        Matrix weights;
        Vector biases;
        Vector errors;

        ActivationFunctions::ActivationFunction* activationFunction;

        DenseLayer *previous;
        DenseLayer *next;
        Vector activations;
        Vector weightedInputs;

        DenseLayer(int inputSize, int neurons);
        DenseLayer(int inputSize, int neurons, ActivationFunctions::ActivationFunction* activationFunction);
        ~DenseLayer();
        void getOutput() override;
        void print() override;
        void printOutput();
        void getError();
    };

    DenseLayer::~DenseLayer() {
        delete this->activationFunction;
    }

    DenseLayer::DenseLayer(int inputSize, int neurons)
    : weights(Matrix::Random(inputSize, neurons)), biases(Vector::Random(neurons)),
              previous(nullptr), next(nullptr), activations(Vector(neurons)), errors(Vector(neurons)), neurons(neurons),
              weightedInputs (Vector(neurons)){
        this->activationFunction = new ActivationFunctions::identity();
    }

    DenseLayer::DenseLayer(int inputSize, int neurons, ActivationFunctions::ActivationFunction* activationFunction)
    : DenseLayer(inputSize, neurons){
        this->activationFunction = activationFunction;
    }

    void DenseLayer::getOutput() {
        for (int i = 0; i < this->weights.cols(); i++) {
            //Calculates the activations of the previous array times the weights for each neuron
            this->activations(i) = (this->previous->activations.array() * this->weights.col(i).array()).sum();
            //Adds the bias
            this->activations(i) += biases(i);
            //Copies the weighted inputs
            this->weightedInputs(i) = this->activations(i);
            //Applies the activation function
            this->activations(i) = this->activationFunction->activation(this->activations(i));
        }
    }

    void DenseLayer::print(){
        std::cout << "Weights:\n" << this->weights << "\n";
        std::cout << "Biases:\n" << this->biases << "\n";
        std::cout << "Activations:\n" << this->activations << "\n";
        std::cout << "Errors:\n" << this->errors << "\n";
    }

    void DenseLayer::printOutput(){
        std::cout << "Input:\n" << this->activations << "\n";
    }

    void DenseLayer::getError(){

    }
}









