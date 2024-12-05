#include "nn.h"

Neuron::Neuron(int numInputs) {
  weights.reserve(numInputs);
  for (int i = 0; i < numInputs; ++i) {
    weights.push_back(new Value(getRandomNumber_()));
  }

  for (int i = 0; i < numInputs; ++i) {
    weights[i]->setLabel("weight_" + std::to_string(i));
  }

  bias = Value(getRandomNumber_());
  bias.setLabel("bias");

  products_.reserve(numInputs);
  accSums_.reserve(numInputs - 1);
}

Neuron::~Neuron() {
  int n = weights.size();
  for (int i = 0; i < weights.size() - 1; ++i) {
    delete weights[i];
    delete products_[i];
    delete accSums_[i];
  }

  delete weights[n - 1];
  delete products_[n - 1];
}

Value Neuron::forward(std::vector<Value*>& inputs) {
  products_.clear();
  products_.resize(weights.size());
  accSums_.clear();
  accSums_.resize(weights.size() - 1);
  for (int i = 0; i < inputs.size(); ++i) {
    products_[i] = new Value(*weights[i] * (*inputs[i]));
  }

  for (int i = 0; i < inputs.size(); ++i) {
    products_[i]->setLabel("product_" + std::to_string(i));
  }

  accSums_[0] = new Value(*products_[0] + *products_[1]);
  for (int i = 1; i < inputs.size() - 1; ++i) {
    accSums_[i] = new Value(*accSums_[i - 1] + *products_[i + 1]);
  }

  for (int i = 0; i < inputs.size() - 1; ++i) {
    accSums_[i]->setLabel("accSum_" + std::to_string(i));
  }

  Value output = bias + *accSums_[accSums_.size() - 1];
  output.setLabel("output");
  
  return output;
}

int Neuron::nParams() {
  return weights.size() + 1;
}

double Neuron::getRandomNumber_() {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<> distrib(-1.0, 1.0);
  return distrib(gen);
}

Layer::Layer(int numInputs, int numNeurons) {
  neurons.reserve(numNeurons);
  for (int i = 0; i < numNeurons; ++i) {
    neurons.emplace_back(numInputs);
  }
}

std::vector<Value*> Layer::forward(std::vector<Value*>& inputs) {
  std::vector<Value*> output;
  output.reserve(neurons.size());
  for (auto& n : neurons) {
    output.push_back(new Value(n.forward(inputs)));
  }

  return output;
}

int Layer::nParams() {
  int result = 0;
  for (auto& n : neurons) {
    result += n.nParams();
  }

  return result;
}

MLP::MLP(int numInputs, const std::vector<int>& numOutputs) {
  int numLayers = numOutputs.size() + 1;
  layers.reserve(numLayers);

  layers.emplace_back(numInputs, numOutputs[0]); // input layer

  // hidden layers and output layer
  for (int i = 0; i < numOutputs.size() - 1; ++i) {
    layers.emplace_back(numOutputs[i], numOutputs[i + 1]);
  }
}

std::vector<Value*> MLP::forward(std::vector<Value*>& inputs) {
  std::vector<Value*> outputs = inputs;
                       
  for (auto& layer : layers) {
    outputs = layer.forward(outputs);
  }
  return outputs;
}

int MLP::nParams() {
  int result = 0;
  for (auto& l : layers) {
    result += l.nParams();
  }
  return result;
}
