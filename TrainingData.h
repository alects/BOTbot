#ifndef TrainingData_H_
#define TrainingData_H_

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class TrainingData
{
public:
  TrainingData(const std::string filename);
  bool isEof() { return m_trainingDataFile.eof(); }
  void getTopology(std::vector<unsigned> &topology);

  unsigned getNextInputs(std::vector<double> &inputVals);
  unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
  std::ifstream m_trainingDataFile;
};
#endif
