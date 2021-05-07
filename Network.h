#include <vector>
#include "Layer.h"
#pragma once

namespace ExperiNet{
    class AbstractNeuralNetwork{
        std::vector<AbstractLayer> layers;
        int inputSize;
        int outputSize;
    };
}