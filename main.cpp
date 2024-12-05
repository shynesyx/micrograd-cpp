#include <string>
#include <vector>
#include "nn.h"

int main() {
  Value a(1.0);
  a.setLabel("a");
  Value b(2.0);
  b.setLabel("b");
  Value c(10.0);
  c.setLabel("c");
  Value d = a * b;
  d.setLabel("d");
  Value e = c + d;
  e.setLabel("e");

  std::cout << &a << ","
            << &b << ","
            << &c << ","
            << &d << ","
            << &e << std::endl;

  std::vector<Value*> f;
  f.resize(3);
  for (int i = 0; i < 3; ++i) {
    f[i] = new Value(20.0 + (double)i);
    f[i]->setLabel("f" + std::to_string(i));
  }

  std::vector<Value*> g;
  g.resize(3);
  for (int i = 0; i < 3; ++i) {
    Value* left;
    if (i == 0) {
      left = &e;
    } else {
      left = g[i - 1];
    }
    
    g[i] = new Value(*left + *f[i]);
    g[i]->setLabel("g" + std::to_string(i));
  }

  std::cout << "----------------\nBack Propagation" << std::endl;
  e.backward(true);
  // zero-grad before back-propogation every time
  // or grad values will carry over
  a.setGrad(0.0);
  b.setGrad(0.0);
  c.setGrad(0.0);
  d.setGrad(0.0);
  e.setGrad(0.0);
  g[2]->backward(true);
  
  // clean up
  for (int i = 0; i < 3; ++i) {
    delete f[i];
    delete g[i];
  }
  f.clear();
  g.clear();
  
  std::cout << "----------------\nNeuron" << std::endl;
  Neuron n1(3);
  std::vector<Value*> n1Inputs{new Value(1.0), new Value(2.0), new Value(3.0)};
  for (int i = 0; i < 3; ++i) {
    n1Inputs[i]->setLabel("x" + std::to_string(i));
  }
  Value n1Output = n1.forward(n1Inputs);
  n1Output.setLabel("output");
  n1Output.backward(true);
  
  for (int i = 0; i < 3; ++i) {
    delete n1Inputs[i];
  }
  
  n1Inputs.clear();
  
  std::cout << "----------------\nLayer" << std::endl;
  Layer l1(3, 1); // 3 inputs; 3 neurons; should return 12 parameters
  std::cout << "Number of neurons: " << l1.neurons.size() << std::endl;
  for (auto& n : l1.neurons) {
    std::cout << "  Number of parameters: " << n.nParams() << std::endl;
    for (auto& w : n.weights) {
      std::cout << w->getData() << std::endl;
    }
    std::cout << n.bias.getData() << std::endl;
  }

  std::cout << "Total number of parameters: " << l1.nParams() << std::endl;

  std::vector<Value*> l1Inputs {new Value(1.0), new Value(2.0), new Value(3.0)};
  for (int i = 0; i < 3; ++i) {
    l1Inputs[i]->setLabel("x" + std::to_string(i));
  }
  std::vector<Value*> l1Outputs = l1.forward(l1Inputs);

  l1Outputs[0]->backward();

  for (int i = 0; i < 3; ++i) {
    delete l1Inputs[i];
  }

  l1Inputs.clear();

  std::cout << "----------------\nMLP" << std::endl;
  std::vector<int> numOutputs{4, 4, 1};
  MLP mlp(3, numOutputs);

  std::cout << "Number of layers: " << mlp.layers.size() << std::endl;
  for (auto& l : mlp.layers) {
    std::cout << "  Number of neurons: " << l.neurons.size() << std::endl;
    for (auto& n : l.neurons) {
      std::cout << "    Number of parameters: " << n.nParams() << std::endl;
      for (auto& w : n.weights) {
        std::cout << w->getData() << std::endl;
      }
      std::cout << n.bias.getData() << std::endl;
    }
  }

  std::cout << "Total number of parameters: " << mlp.nParams() << std::endl;


  std::cout << "----------------\nGradient Descent" << std::endl;
  // step 0: define data and hyperparameters
  const int N_ROWS = 4; // number of observations
  const int N_COLS = 3; // number of features
  std::vector< std::vector<double> > values{
    {2.0,  3.0, -1.0},
    {3.0, -1.0,  0.5},
    {0.5,  1.0,  1.0},
    {1.0,  1.0, -1.0}
  };
  std::vector< std::vector<Value*> > mlpInputs(N_ROWS, std::vector<Value*>(N_COLS));
  for (int i = 0; i < N_ROWS; ++i) {
    for (int j = 0; j < N_COLS; ++j) {
      mlpInputs[i][j] = new Value(values[i][j]);
      mlpInputs[i][j]->setLabel("input_" + std::to_string(i) + "_" + std::to_string(j));
    }
  }

  std::vector<double> targetValues{1.0, -1.0, -1.0, 1.0};
  std::vector<Value*> targets(N_ROWS);
  for (int i = 0; i < N_ROWS; ++i) {
    targets[i] = new Value(targetValues[i]);
    targets[i]->setLabel("target_" + std::to_string(i));
  }
  
  // hyperparameters
  double alpha = 0.01; // learning rate
  int numEpoch = 100;  // number of iterations

  std::vector<Value*> predictions(N_ROWS);

  for (int epoch = 0; epoch < numEpoch; ++epoch) {
    // step 1: calculate loss
    for (int i = 0; i < N_ROWS; ++i) {
      predictions[i] = mlp.forward(mlpInputs[i])[0];
      predictions[i]->setLabel("prediction_" + std::to_string(i));
    }


    std::vector<Value*> diffs(N_ROWS);
    std::vector<Value*> diffSquares(N_ROWS);
    std::vector<Value*> diffSquareAccSums(N_ROWS - 1);
    for (int i = 0; i < N_ROWS; ++i) {
      diffs[i] = new Value(*predictions[i] - *targets[i]);
      diffs[i]->setLabel("diff_" + std::to_string(i));
      diffSquares[i] = new Value((*diffs[i]) * (*diffs[i]));
      diffSquares[i]->setLabel("diffSquare_"+std::to_string(i));
    }

    diffSquareAccSums[0] = new Value(*diffSquares[0] + *diffSquares[1]);
    diffSquareAccSums[0]->setLabel("diffSquareAccSum_0");
    for (int i = 1; i < N_ROWS - 1; ++i) {
      diffSquareAccSums[i] = new Value(*diffSquareAccSums[i - 1] + *diffSquares[i + 1]);
      diffSquareAccSums[i]->setLabel("diffSquareAccSum_" + std::to_string(i));
    }


    // step 2: back propagate gradients
    diffSquareAccSums[N_ROWS - 2]->backward(); // the final loss

    // step 3: adjust weights and biases
    for (auto& l : mlp.layers) {
      for (auto& n : l.neurons) {
        for (auto& w : n.weights) {
          w->setData(w->getData() - alpha * w->getGrad());
          w->setGrad(0.0); // make sure to "zero grad"
        }
        n.bias.setData(n.bias.getData() - alpha * n.bias.getGrad());
        n.bias.setGrad(0.0); // make sure to "zero grad";
      }
    }

    // step 4: repeat
  }

  // print out the predicted values
  for (int i = 0; i < N_ROWS; ++i) {
    std::cout << predictions[i]->getData() << ", ";
  }
  std::cout << std::endl;

  // clean up
  for (int i = 0; i < N_ROWS; ++i) {
    delete targets[i];
    delete predictions[i];
    for (int j = 0; j < N_COLS; ++j) {
      delete mlpInputs[i][j];
    }
  }
  
  return 0;
}
