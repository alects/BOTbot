//TODO:
//Questions: struct in header file?
//function def'n (randomWeight) in header file?

#ifndef NNET_H_
#define NNET_H_


struct Connection
{
  double weight;
  double delta_weight;
};

//The input layer made up of input vector pixel values + a bias neuron.
class Neuron;

//Connection struct represents every forward connection
//between neurons in adjacent layers.
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
  Neuron(unsigned numOutputs, unsigned indx);

  double getOutputVal() const{ return outVal; }
  void setOutputVal(double x) { outVal = x; }
  void feedForward(const Layer &prevLayer);
  void calcOutGrads(double targetVal);
  void updateInputWeights(Layer &prevLayer);
  void calcHidGrads(const Layer &nextLayer);

private:
  //alpha is nonzero value to keep neurons alive ;)
  const double alpha = .001;
  //beta is the momentum: 0- none, .5-moderate
  const double beta = .1;
  //omega is the learning rate: 0- slow, 1- reckless
  const double omega = .001;

  double activation(double x);
  double activationDer(double x);
  //random number [0, 1]
  static double randomWeight() { return rand() / double(RAND_MAX); }
  double sumDOW(const Layer &nextLayer) const;

  double outVal;
  double inVal;
  std::vector<Connection> outWeights;
  unsigned index;
  unsigned gradient;
};

class Net
{
public:
  Net(const std::vector<unsigned> &topology);
  void feedForward(const std::vector<double> &inputVals);
  void backProp(const std::vector<double> &targetVal);
  void getResults(std::vector<double> &resultVals) const;
  double getAveErr() const { return aveErr; }

private:
  std::vector<Layer> layers;
  double err;
  double aveErr;
  double aveErrSmF;
};
#endif
