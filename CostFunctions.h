#pragma once
#include <math.h>

namespace ExperiNet{
    float crossEntropy(float output, float truth){
        //remember to divide by n, the items of training data
        return -(truth * std::log(output) + (1 - truth) * std::log(1 - output));
    }
}

