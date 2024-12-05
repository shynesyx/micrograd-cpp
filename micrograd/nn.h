#ifndef NN_H
#define NN_H

#include <memory>
#include <numeric>
#include <random>
#include <string>
#include "engine.h"

class Neuron {
public:
  std::vector<Value*> weights;
  Value bias;

  Neuron(int);

  ~Neuron();

  Value forward(std::vector<Value*>&);

  int nParams();

private:
  std::vector<Value*> products_; // pair-wise products of inputs and weights
  std::vector<Value*> accSums_;  // accumulated sum of products_
  double getRandomNumber_();
};

class Layer {
public:
  std::vector<Neuron> neurons;

  Layer(int, int);

  std::vector<Value*> forward(std::vector<Value*>&);

  int nParams();
};

class MLP {
public:
  std::vector<Layer> layers;

  MLP(int, const std::vector<int>&); // numberInputs, [numberNeurons]

  std::vector<Value*> forward(std::vector<Value*>&);

  int nParams();
};


#endif // NN_H
