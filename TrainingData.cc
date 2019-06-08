#include "TrainingData.h"


void TrainingData::getTopology(std::vector<unsigned> &topology)
{
  std::string line;
  std::string label;

  getline(m_trainingDataFile, line);
  std::stringstream ss(line);
  ss >> label;
if(this->isEof() || label.compare("topology:") != 0) {}

  while(!ss.eof()) {
    unsigned x;
    ss >> x;
    topology.push_back(x);
  }
  return;
}

TrainingData::TrainingData(const std::string filename)
{
  m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
  inputVals.clear();

  std::string line;
  getline(m_trainingDataFile, line);
  std::stringstream ss(line);

  std::string label;
  ss >> label;
  if(label.compare("in:") == 0) {
    double oneVal;
    while(ss >> oneVal) {
      inputVals.push_back(oneVal);
    }
  }
  return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
{
  targetOutputVals.clear();

  std::string line;
  getline(m_trainingDataFile, line);
  std::stringstream ss(line);

  std::string label;
  ss >> label;
  if(label.compare("out:") == 0) {
    double oneVal;
    while(ss >> oneVal) {
      targetOutputVals.push_back(oneVal);
    }
  }
  return targetOutputVals.size();
}
