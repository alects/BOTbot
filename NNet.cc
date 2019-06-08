//TODO:
//Description below may be better in readfile or something, for
//now it's just to help understand abstract process for coding.. .
//Remove redundant libraries.
//Declare bias as const in header (not hardcoded)
//Break-up further into sep neuron and net source/headers?
//
//BOTbot: A Tree Classification Neural Network
//By: Brendon Kieser, Irfan Filipovic, Alexander Petzold
//For: CIS 330 C & C++
//Date: May 2019
//
//BOTbot is a neural network that classifies tree family
//based on leaf shape.  The neural network frame is based loosely
//off of David Miller's Neural Net in C++ Tutorial: vimeo.com//19569529.
//
//Training
//1. Training data set is made up of images of leaves contained in color_img
//folder.  Note: The photos do not have to be colored.
//2. Each photo is converted into a black/white and exported to bw_img/.
//3. Leaf edges are calculated using opencv and converted to binary values
//0-black 1-white copied into an array of vector<int> type.
//4. Vectors are stored with target family name and unique identifier.
//5. Target classification is extracted from each data point.
//6. Training data is run through the network with the target classification.
//to attain weights and bias parameters.
//
//Classification
//1. Image to be classified is placed source containing folder and passed as
//argument in command line.
//2. The image is pre-processed as in steps 2-3.
//3. Output is predicted family name and certainty percentage.
//

//Code used to train/test dataset (re-place in main.cc)
/*
int main() {
  srand(0);
  const std::string SOURCE = "edge_img/*.bmp";
  // get images from source folder
  std::vector<std::string> folder;
  std::vector<cv::String> folder_cv;
  cv::glob(SOURCE, folder_cv);
  for (auto &i : folder_cv)
    folder.push_back(i);
  int photo_count = folder.size();
  std::vector<unsigned> topology = {18639, 3, 10};
  Net nNet(topology);
  // Loop through every image
  for (int k = 0; k < photo_count; ++k) {
    // Read the image file
    std::string leaf = folder[k];
    cv::Mat img = cv::imread(leaf, 0);
    std::cout << leaf << "\n";

    std::vector<double> targets(10);
    targets[std::atoi(&leaf[9])] = 1.0;

    // Check for file read failure
    if (img.empty()) {
      std::cout << "oops" << std::endl;
      std::cin.get();
      return 1;
    }

    //Make a vector from the cv::Mat object
    std::vector<double> v, resultVals;
    //get the dimensions
    cv::Size s = img.size();
    int rows = s.height;
    int cols = s.width;
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        v.push_back(img.at<uchar>(i,j));
      }
    }
    nNet.feedForward(v);
    nNet.getResults(resultVals);
    printVals("Outputs", resultVals);

    printVals("Targets", targets);

    nNet.backProp(targets);

    std::cout << nNet.getAveErr() << "\n";
  }

  return 0;
}
*/

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

#include "NNet.h"

double Neuron::sumDOW(const Layer &nextLayer) const
{
  double sum = 0.0;
  double n_size = nextLayer.size() - 1;
  //Getting the sum of all the errors at the given node
  for(unsigned i = 0; i < n_size; ++i) {
    sum += outWeights[i].weight * nextLayer[i].gradient;
  }
  return sum;
}

void Neuron::calcOutGrads(double targetVal)
{
  double delta = targetVal - outVal;
  gradient = delta * Neuron::activationDer(inVal);
}

void Neuron::calcHidGrads(const Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);
  gradient = dow * Neuron::activationDer(inVal);
}

void Neuron::updateInputWeights(Layer &prevLayer)
{
  double p_size = prevLayer.size();

  for(unsigned i = 0; i < p_size; ++i) {
    Neuron &neuron = prevLayer[i];
    double oldDelta = neuron.outWeights[index].delta_weight;
    double newDelta = omega * neuron.getOutputVal() * gradient + beta * oldDelta;

    //Save the changes to weight and update weight.
    neuron.outWeights[index].delta_weight = newDelta;
    neuron.outWeights[index].weight += newDelta;
  }
}

double Neuron::activation(double x)
{
  //Leaky reLu
  return (x < 0) ? alpha * x : x;
}

double Neuron::activationDer(double x)
{
  return (x < 0) ? alpha : 1;
}

void Neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;

  for(unsigned i = 0; i < prevLayer.size(); ++i) {
    sum += prevLayer[i].getOutputVal() * prevLayer[i].outWeights[index].weight;
  }

  //shape output using activation function
  inVal = sum;
  outVal = Neuron::activation(sum);

}

Neuron::Neuron(unsigned numOutputs, unsigned indx)
{
  for(unsigned c = 0; c < numOutputs; ++c) {
    outWeights.push_back(Connection());
    //setting random weights
    outWeights.back().weight = randomWeight();
  }
  index = indx;
}


void Net::getResults(std::vector<double> &resultVals) const
{
  resultVals.clear();

  for(unsigned i = 0; i < layers.back().size(); ++i) {
    resultVals.push_back(layers.back()[i].getOutputVal());
  }
}

void Net::backProp(const std::vector<double> &targetVal)
{
  //Calculating errors (RMS)
  Layer &outputLayer = layers.back();
 err= 0.0;

  double oLsize = outputLayer.size() - 1;
  double m_Lsize = layers.size();

  for(unsigned i = 0; i < oLsize; ++i) {
    double delta = targetVal[i] - outputLayer[i].getOutputVal();
   err+= delta * delta;
  }
  err/= oLsize;
  err= sqrt(err);

   aveErr = (aveErr * aveErrSmF + err) / (aveErrSmF + 1.0);

  //Calculate Layer gradients
  for(unsigned int i = 0; i < oLsize; ++i) {
    outputLayer[i].calcOutGrads(targetVal[i]);
  }

  //Calculate hidden Layer gradient
  for(unsigned i = m_Lsize - 2; i > 0; --i) {
    Layer &hiddenLayer = layers[i];
    Layer &nextLayer = layers[i + 1];
    double h_lsize = hiddenLayer.size();

    for(unsigned j = 0; j < h_lsize; ++j) {
      hiddenLayer[j].calcHidGrads(nextLayer);
    }
  }

  //Update all connection weights
  for(unsigned i = m_Lsize - 1; i > 0; --i){
    Layer &layer = layers[i];
    Layer &prevLayer = layers[i - 1];
    double l_size = layer.size();

    for(unsigned j = 0; j < l_size - 1; ++j) {
      layer[j].updateInputWeights(prevLayer);
    }
  }
}

void Net::feedForward(const std::vector<double> &inputVals)
{
  //number of elements input == number of neurons
  assert(inputVals.size() == layers[0].size() - 1);

  for(unsigned i = 0; i < inputVals.size(); ++i) {
    layers[0][i].setOutputVal(inputVals[i]);
  }

  //forward propogate
  for(unsigned int i = 0; i < layers.size(); ++i) {
    Layer &prevLayer = layers[i - 1];
    for(unsigned int j = 0; j < layers[i].size() - 1; ++j) {
      layers[i][j].feedForward(prevLayer);
    }
  }
}

//MAKE CLASS NEURON AND CLASS NET FRIENDS??

Net::Net(const std::vector<unsigned> &topology)
{
  unsigned numlayers = topology.size();
  //Creating layers
  for(unsigned i = 0; i < numlayers; ++i) {
    layers.push_back(Layer());

    //Adding the Neurons and the bias
    unsigned numOutputs = i == topology.size() -1 ? 0 : topology[i + 1];
    for(unsigned j = 0; j <= topology[i]; ++j) {
      layers.back().push_back(Neuron(numOutputs, j));
      std::cout << "neuron added" << std::endl;
    }

    //set bias to 1.0
    layers.back().back().setOutputVal(1.0);
  }
}
