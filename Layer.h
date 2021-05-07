#pragma once
#include <eigen-3.3.9/Eigen/Dense>
#include "ActivationFunctions.h"

namespace ExperiNet{
    typedef Eigen::MatrixXf Matrix;
    typedef Eigen::VectorXf Vector;

    class AbstractLayer{
        virtual Vector getOutput() = 0;
    };

    class DenseLayer : AbstractLayer{
    public:
        Matrix weights;
        Vector biases;
        float (*activation)(float);
        DenseLayer* previous;
        DenseLayer* next;
        Vector output;
        DenseLayer(int inputSize, int neurons);
        DenseLayer(int inputSize, int neurons, float (*activation)(float));
        Vector getOutput() override;
    };

    DenseLayer::DenseLayer(int inputSize, int neurons)
            : weights(Matrix(inputSize, neurons)), biases(Vector(neurons)), activation(&lrelu),
              previous(nullptr), next(nullptr), output(Vector(neurons)) {}

    DenseLayer::DenseLayer(int inputSize, int neurons, float (*activation)(float))
            : weights(Matrix(inputSize, neurons)), biases(Vector(neurons)), activation(activation),
              previous(nullptr),next(nullptr), output(Vector(neurons)) {}

    Vector DenseLayer::getOutput() {
        for (int i = 0; i < this->weights.cols(); i++){
            //Calculates the output of the previous array times the weights for each neuron
            this->output(i) = (this->previous->output.array() * this->weights.col(i).array()).sum();
            //Adds the bias
            this->output(i) += biases(i);
            //Applies the activation function
            this->output(i) = (float) activation((float) this->output(i));
            std::cout << this->output << std::endl;
        }
        return output;
    }
}









