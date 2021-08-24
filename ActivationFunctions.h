#pragma once

namespace ExperiNet{
    namespace ActivationFunctions{

        class ActivationFunction{
        public:
            virtual float activation(float x) = 0;
            virtual float derivative(float x) = 0;
            virtual ~ActivationFunction(){};
        };

        class tanh : public ActivationFunction{
        public:
            float activation(float x) override{
                return (float) std::tanh((double) x);
            }
            float derivative(float x) override{
                return 1;
            }
        };
        class identity : public ActivationFunction{
        public:
            float activation(float x) override{
                return x;
            }
            float derivative(float x) override{
                return 1;
            }
        };
        class relu : public ActivationFunction{
        public:
            float activation(float x) override{
                return x > 0 ? x : 0;
            }
            float derivative(float x) override{
                return x > 0 ? 1 : 0;
            }
        };
        class lrelu : public ActivationFunction{
        public:
            float activation(float x) override{
                return x > 0 ? x : 0.1 * x;
            }
            float derivative(float x) override{
                return x > 0 ? 1 : 0.1;
            }
        };
    }
}