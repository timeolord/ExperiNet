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
        Matrix weights;
        Vector biases;

        float (*activation)(float);

        DenseLayer *previous;
        DenseLayer *next;
        Vector output;

        DenseLayer(int inputSize, int neurons);

        DenseLayer(int inputSize, int neurons, float (*activation)(float));

        void getOutput() override;

        void print() override;

        void printOutput();
    };

    DenseLayer::DenseLayer(int inputSize, int neurons)
            : weights(Matrix::Random(inputSize, neurons)), biases(Vector::Random(neurons)), activation(&none),
              previous(nullptr), next(nullptr), output(Vector(neurons)) {
    }

    DenseLayer::DenseLayer(int inputSize, int neurons, float (*activation)(float))
            : weights(Matrix::Random(inputSize, neurons)), biases(Vector::Random(neurons)), activation(activation),
              previous(nullptr), next(nullptr), output(Vector(neurons)) {
    }

    void DenseLayer::getOutput() {
        for (int i = 0; i < this->weights.cols(); i++) {
            //Calculates the output of the previous array times the weights for each neuron
            this->output(i) = (this->previous->output.array() * this->weights.col(i).array()).sum();
            //Adds the bias
            this->output(i) += biases(i);
            //Applies the activation function
            this->output(i) = (float) activation((float) this->output(i));
        }
    }

    void DenseLayer::print(){
        std::cout << "Weights:\n" << this->weights << "\n";
        std::cout << "Biases:\n" << this->biases << "\n";
        std::cout << "Output:\n" << this->output << "\n";
    }

    void DenseLayer::printOutput(){
        std::cout << "Input:\n" << this->output << "\n";
    }
}









