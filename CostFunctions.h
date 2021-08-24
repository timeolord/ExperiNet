#pragma once
#include <math.h>

namespace ExperiNet{
    namespace CostFunctions{
        class CostFunction{
        public:
            virtual float cost(float output, float truth) = 0;
            virtual float derivative(float output, float truth) = 0;
            virtual ~CostFunction(){};
        };
        class MSE : public CostFunction{
        public:
            //remember to divide by n, the items of training data
            float cost(float output, float truth) override{
                return (truth - output) * (truth - output);
            }
            float derivative(float output, float truth) override{
                return (truth - output);
            };
        };
    }
}

