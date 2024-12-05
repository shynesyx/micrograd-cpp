#include "engine.h"
#include <functional>

// Constructors
// grad_ initialized to 0.0 because as it propagates, it will accumulate
Value::Value(double data, std::set<Value*> children, std::string op)
  : data_(data), prev_(children), op_(op), grad_(0.0), backward_([]() {}) {}

Value::Value(double data) : Value::Value(data, {}, "") {}

// Copy constructor
Value::Value(const Value &other)
  : data_(other.data_), grad_(other.grad_), prev_(other.prev_),
    op_(other.op_), label_(other.label_), backward_(other.backward_) {}

// Operator overloading
Value Value::operator+(Value &other) {
  Value out = helper_(data_ + other.data_, other, "+");

  out.backward_ = [this, &other, &out]() {
    // std::cout << ">> updating grad (" << out.grad_ << ","
    //           << this << "," << &other << "," << &out << "); ";
    this->grad_ += out.grad_;
    other.grad_ += out.grad_;
  };

  return out;
}

Value Value::operator-(Value& other) {
  Value out = helper_(data_ - other.data_, other, "-");

  out.backward_ = [this, &other, &out]() {
    this->grad_ += out.grad_;
    other.grad_ -= out.grad_;
  };

  return out;
}

Value Value::operator*(Value& other) {
  Value out = helper_(data_ * other.data_, other, "*");

  out.backward_ = [this, &other, &out]() {
    this->grad_ += other.data_ * out.grad_;
    other.grad_ += this->data_ * out.grad_;
  };

  return out;
}

Value Value::operator/(Value& other) {
  Value out = helper_(data_ / other.data_, other, "/");

  out.backward_ = [this, &other, &out]() {
    double otherInv = 1.0 / other.data_;
    this->grad_ += otherInv * out.grad_;
    other.grad_ += -this->data_ * otherInv * otherInv * out.grad_;
  };

  return out;
}

// bool Value::operator<(const Value& other) const {
//   return data_ < other.data_;
// }

// Value& operator=(const Value & other) {
//   data_ = other.data_;
//   grad_ = other.grad_;
//   prev_ = other.prev_;

//   label_ = other.label_;
//   op_ = other.op_;

//   return *this;
// }

// Public member functions
void Value::backward(bool shouldPrintTopo = false) {
  std::vector<const Value *> topo;
  std::set<const Value *> visited;

  buildTopo_(*this, topo, visited);

  // Print topology if requested
  if (shouldPrintTopo) {
    for (const auto& el : topo) {
      std::cout << el->getLabel();
      std::cout << "; " << el << ";";
      if (!el->prev_.empty()) {
        std::cout << " (";
        for (Value *const &c : el->prev_) {
          std::cout << c->getLabel() << " " << c << ",";
        }
        std::cout << "\b)";
      }
      std::cout << std::endl;
    }
  }

  // std::cout << "Topo has " << topo.size() << " nodes" << std::endl;

  // Back-propagate through all nodes
  grad_ = 1; // the first node in topology whose derivative wrt itself is equal to 1
  for (int i = topo.size() - 1; i >= 0; --i) {
    // std::cout << "index = " << i << "; " << topo[i]->getLabel() << "; ";
    try {
      topo[i]->backward_();
    } catch (const std::bad_function_call& e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    // std::cout << "grad = " << topo[i]->getGrad() << std::endl;
  }
}

void Value::backward() { backward(false); }

// Setters
void Value::setData(double value) { data_ = value; }

void Value::setGrad(double value) { grad_ = value; }

void Value::setLabel(std::string label) { label_ = label; }

// Private utility functions
Value Value::helper_(double data, Value &other, std::string op) {
  std::set<Value*> children{this, &other};
  return Value(data, children, op);
}

void Value::buildTopo_(const Value& node, std::vector<const Value*>& topo, std::set<const Value*>& visited) {
  // Together with the `topo` and `visited` variables, the code below
  // implements the *topological sort* for a DAG
  const bool hasVisited = visited.find(&node) != visited.end();
  if (!hasVisited) {
    visited.insert(&node);
    for (Value *const &n : node.prev_) {
      buildTopo_(*n, topo, visited);
    }
    topo.push_back(&node);
  }
}
