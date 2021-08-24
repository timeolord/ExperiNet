#pragma once

namespace ExperiNet{
    namespace ActivationFunctions{

        class ActivationFunction{
        public:
            virtual float activation(float x) = 0;
            virtual float derivative(float x) = 0;
            virtual std::string name() = 0;
            virtual ~ActivationFunction(){};
        };

        class tanh : public ActivationFunction{
        public:
            float activation(float x) override{
                return (float) std::tanh((double) x);
            }
            float derivative(float x) override{
                return 1 - (activation(x) * activation(x));
            }
            std::string name() override{
                return "tanh";
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
            std::string name() override{
                return "identity";
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
            std::string name() override{
                return "relu";
            }
        };
        class sigmoid : public ActivationFunction{
        public:
            float activation(float x) override{
                return 1 / (1 + std::exp(-x));
            }
            float derivative(float x) override{
                return 1 - activation(x);
            }
            std::string name() override{
                return "sigmoid";
            }
        };
        class positiveTanh : public ActivationFunction{
        public:
            float activation(float x) override{
                return (float) ((std::tanh((double) x) + 1) / 2);
            }
            float derivative(float x) override{
                return 1 - (activation(x) * activation(x));
            }
            std::string name() override{
                return "positive tanh";
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
            std::string name() override{
                return "lrelu";
            }
        };
    }
}