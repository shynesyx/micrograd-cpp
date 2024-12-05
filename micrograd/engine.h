#ifndef ENGINE_H
#define ENGINE_H

#include <set>
#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <fstream>

class Value
{
public:

  // Constructors
  Value() : data_(0.0), grad_(0.0) {};
  Value(double);
  Value(double, std::set<Value*>, std::string);


  // Copy constructor
  Value(const Value&);

  // Destructor
  ~Value() {};

  // Overloading operators

  Value operator+(Value &);
  Value operator-(Value &);
  Value operator*(Value &);
  Value operator/(Value &);
  // bool  operator<(const Value &) const;
  // Value& operator=(const Value &);

  // Member functions
  void backward(bool);
  void backward();

  // Getters
  double getData() const { return data_; }
  std::set<Value*> getPrev() const { return prev_; }
  double getGrad() const { return grad_; }
  std::string getOperator() const { return op_; }
  std::string getLabel() const { return label_; }

  // Setters
  void setData(double);
  void setGrad(double);
  void setLabel(std::string);

private:
  double data_;
  double grad_;
  std::string label_;
  std::set<Value*> prev_;
  std::string op_;

  std::function<void()> backward_;

  Value helper_(double, Value&, std::string);
  void buildTopo_(const Value&, std::vector<const Value *>&, std::set<const Value *>&);
};

#endif // ENGINE_H
