#pragma once

namespace ExperiNet{
    float tanh (float x){
        return (float) std::tanh((double) x);
    }

    float relu (float x){
        return x > 0 ? x : 0;
    }

    float lrelu (float x){
        return x > 0 ? x : 0.01 * x;
    }
}