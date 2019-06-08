// Very basic CNN, with forward prop. Need to implement backProp and getResult
// Clean up functions and improve header file

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include "CNN.h"
#include <time.h>
#include <float.h>

// Convert single dimension input picture, to 2D std::vector
std::vector<std::vector<double>> CNN::to2D(std::vector<short> in) {
  // Create out 2D std::vector, create 1D std::vector to use as filler
  std::vector<std::vector<double>> out(maxRow);
  std::vector<double> nest(maxCol);
  // Loop through each Row of input
  for(int i = 0; i < maxRow; ++i) {
    // Loop through each Col of input
    for(int j = 0; j < maxCol; ++j) {
      // Set nest = to matching std::vector in input
      nest[j] = in[(i * maxCol) + j];
    }
    // Set nest in correct row of 2D
    out[i] = nest;
  }
  // Return 2D std::vector
  return out;
}

double CNN::dotLeakyReLU(int i, int j,
                         std::vector<std::vector<double>> in,
                         std::vector<std::vector<double>> w) {
  // Initialize vars that will be used
  int tmpi = 0, tmpj = 0;
  double sum = 0.0;
  // Leaky ReLU multiplication factor
  double alpha = .001;
  // Loop through each row in 3x3 input
  for(int k = 0; k < filterSize; ++k) {
    // Set temp i
    tmpi = k + i;
    // Loop through each col in 3x3
    for(int l = 0; l < filterSize; ++l) {
      // Set tmpj
      tmpj = l + j;
      // Dot product between weight and in
      sum += ( in[tmpi][tmpj] * w[k][l]);
    }
  }
  // Leaky ReLU
  return (sum < 0.0) ? alpha * sum : sum;
}

double CNN::poolRegion(int i, int j,
                       std::vector<std::vector<double>> in) {
  // Initialize vars that will be used
  int tmpi, tmpj;
  double max = 0;
  // Loop through each row in 2x2 pool filterSize
  for(int k = 0; k < pool; ++k) {
    // Set tmpi
    tmpi = k + (i * pool);
    // Loop for each col in 2x2 pool filterSize
    for(int l = 0; l < pool; ++l) {
      // Set tmpj, val = input
      tmpj = l  + (j * pool);
      double val = in[tmpi][tmpj];
      // If k is 0 and l is 0, initialize max as val;
      if(k == 0 && l == 0) {
        max = val;
      }
      // If val is greater than max, max = val;
      if(max < val) {
        max = val;
      }
    }
  }
  // Return max val for pooling layer
  return max;
}

// Creates the a pooled layer, using pool function
std::vector<std::vector<double>> CNN::poolLayer(std::vector<std::vector<double>> in,
                                                int x, int y) {
  // Initialize empty return std::vector
  std::vector<std::vector<double>> o(y);
	std::vector<double> o1(x);
  for(int i = 0; i < y; ++i) {
  	for(int j = 0; j < x; ++j) {
			o1[j] = 0;
		}
		o[i] = o1;
	}

  // Iterate through in std::vector rows
  for(int i = 0; i < y; ++i) {
    // Iterate through in std::vector cols
    for(int j = 0; j < x; ++j) {
      // Implment pool for each region
      o[i][j] = poolRegion(i, j, in);
    }
  }
  // Return pooled layer
  return o;
}

std::vector<std::vector<double>> CNN::convLayer(std::vector<std::vector<double>> in,
                                                std::vector<std::vector<double>> w,
                                                int x, int y) {
  // Initialize 2D std::vector for return
  std::vector<std::vector<double>> out(y);
  std::vector<double> out1(x);
  for(int i = 0; i < y; ++i) {
    for(int j = 0; j < x; ++j) {
			out1[j] = 0;
		}
  	out[i] = out1;
	}
  // Loop through each row of filterSize regions of input
  for(int i = 0; i < y; ++i) {
    // Loop through each col of filterSize regions of input
    for(int j = 0; j < x; ++j) {
      // Implement convolutional layers upon filterSize region
      out[i][j] = dotLeakyReLU(i, j, in, w);
    }
  }
  // Return output
  return out;
}

// pass through 2 convs then pool
std::vector<std::vector<double>> CNN::convPool(std::vector<std::vector<double>> in,
                                               std::vector<std::vector<double>> w1,
                                               std::vector<std::vector<double>> w2,
                                               int y, int x) {
  // Range for first conv pass
	//std::cout << x << "\n";
	//std::cout << y << "\n";
  int convX1 = x - (filterSize - 1);
  int convY1 = y - (filterSize - 1);
  // Range for second conv pass
  int convX2 = convX1 - (filterSize - 1);
  int convY2 = convY1 - (filterSize - 1);
  // Range for pool pass
  int poolX = ceil(convX2/pool);
  int poolY = ceil(convY2/pool);

	//std::cout << convX1 << ' ' << convY1 << " : " << convX2 << ' ' << convY2 << " : " << poolX << ' ' << poolY << "\n";

  // Create tmp std::vector after first conv Pass
  std::vector<std::vector<double>> tmp = convLayer(in, w1, convX1, convY1);
	//std::cout << "tmp 1: " << "\n";
  // Create tmp1 std::vector after second conv Pass
  std::vector<std::vector<double>> tmp1 = convLayer(tmp, w2, convX2, convY2);
	//std::cout << "tmp 2: " << "\n";
  //std::cout << "\n";
  // Create o std::vector after pool Pass
  std::vector<std::vector<double>> o = poolLayer(tmp1, poolX, poolY);
  // Return output layer
  return o;
}

// Implement 1 conv, and 1 pool
std::vector<std::vector<double>> CNN::convPool(std::vector<std::vector<double>> in,
                                               std::vector<std::vector<double>> w1,
                                               int y, int x) {
  // Range for first conv pass
  int convX1 = x - (filterSize - 1);
  int convY1 = y - (filterSize - 1);
  // Range for pool pass
  int poolX = ceil(convX1/pool);
  int poolY = ceil(convY1/pool);

  // Create tmp1 std::vector after first conv Pass
  std::vector<std::vector<double>> tmp1 = convLayer(in, w1, convX1, convY1);
	//std::cout << "done" << "\n";
  // Create o std::vector after pool pass
  std::vector<std::vector<double>> o = poolLayer(tmp1, poolX, poolY);
	//std::cout << "oof the pool" << "\n";
  // Return output layer
  return o;
}

// Function to fullyConenct 2D std::vector, using 1D std::vector of weights
std::vector<double> CNN::fullyConnect(std::vector<std::vector<double>> in,
                                      std::vector<double> w,
                                      int x, int y) {
  // Initialize std::vector for space x*y
  std::vector<double> connected(10);
  double count = 0;
  // Variable to hold value to place in connected
  //std::ofstream file;
  //file.open("o.txt");
  //file << "\tin[i][j]\tw[k]\tin[i][j] * w[k]\n";
  // Iterate through rows
  for(int k = 0; k < 10; ++k) {
    count = 0;
    // Iterate through rows of weight
    for(int i = 0; i < y; ++i) {
      // Iterate through cols of weight
      for(int j = 0; j < x; ++j) {
        // fullyConnect input into connected output
        count += in[i][j] * w[k];
  //      file << "(" << k << ", " << i << ", " << j
  //                << ")\t" << in[i][j] << "\t"
  //                << w[k] << '\t' << in[i][j] * w[k] << '\n';
      }
  //    file << "\n";
    }
    connected[k] = count;
  }
  //file.close();
  // return connected std::vector
  return connected;
}

// Return random 2D weight std::vector
std::vector<std::vector<double>> CNN::createRandWeights() {
  // Initialize empty weight std::vector, size of filterSize
  // Fill vals with randomWeights[0,1];
  std::vector<std::vector<double>> weights(filterSize);
  std::vector<double> b0(filterSize);

  for(int x = 0; x < filterSize; ++x) {
    for(int y = 0; y < filterSize; ++y) {
      b0[y] = randomWeight();
    }
    weights[x] = b0;
  }
  // return randWeight std::vector<std::vector<>>
  return weights;
}

// Create random weight std::vector for fully connected layer
std::vector<double> CNN::createFCWeights() {
  // Make weight of outputSize, loop and enter rand doubles
  std::vector<double> weights(outputSize);
  for(int i = 0; i < outputSize; ++i) {
    weights[i] = randomWeight();
  }
  // return weights
  return weights;
}

// Print weight vals from 2D std::vector
void CNN::printWeight(std::vector<std::vector<double>> in,
                      int y, int x) {
	std::ofstream myfile;
	myfile.open("text.txt");
	// Loop through each position in weight and print val
  for (int i = 0; i < y; ++i) {
    for(int j = 0; j < x; ++j) {
      myfile << in[i][j] << " ";
    }
    myfile << "\n";
  }
  myfile << "\n";
	myfile << "\n";
	myfile << "\n";
	if(x < 90) {
		myfile.close();
	}
}


// FEEDFORWARD WILL HAVE 3 Pool layers, and 7 Conv layers
// Pass through 2 Conv and pool, then 2 conv and pool, then 3 conv and pool
// Then fully connect, Fully connect will have input of size 9 x 7, filter of 1 x 10

////////////////// CURRENTLY FILTER DEPTH OF 1 to reduce amount of weight std::vectors /////////////

// FeedForward first pass, create random weights then pass through
std::vector<double> CNN::feedForward(std::vector<std::vector<double>> input) {
  // Create weight std::vector for each conv layer: 7 conv layers
  // Create weight std::vector for fullyConnect: 1 fully connect layer
	printWeight(w1, w1.size(), w1[0].size());
  std::vector<std::vector<double>> conv1 = convPool(input, w1, w2,
                                                    input.size(), input[0].size());
  std::vector<std::vector<double>> conv2 = convPool(conv1, w3, w4,
                                                    conv1.size(), conv1[0].size());
  std::vector<std::vector<double>> conv3 = convPool(conv2, w5, conv2.size(),
																										conv2[0].size());
  std::vector<double> fc = fullyConnect(conv3, wFC, conv3.size(), conv3[0].size());

  // normalize output
  int fc_size = fc.size();
  // find min and max
  double max = DBL_MIN;
  double min = DBL_MAX;
  for (int i = 0; i < fc_size; ++i) {
    min = (fc[i] < max) ? fc[i] : max;
    max = (fc[i] > max) ? fc[i] : max;
  }
  for (int i = 0; i < fc_size; ++i) {
    fc[i] = fc[i] / max;
  }

  return fc;
}

// FeedForward, but passed with std::vectors of weights, implement if need. above but pass with weights
std::vector<double> CNN::lossFunc_deriv(std::vector<double> y,
                           std::vector<double> y_hat) {
  std::vector<double> loss(10);
  int y_size = y.size();
  for(int i = 0; i < y_size; ++i) {
    loss[i] = (y_hat[i] - y[i]) * lr;
    std::cout << loss[i]<< "\n";
  }
  return loss;
}

std::vector<std::vector<double>> CNN::updateWeights(std::vector<std::vector<double>> input,
                                                    std::vector<double> loss, double lr) {
  //double prod = loss * lr;
  int in_size = input.size();
  int inner_size = input[0].size();
  for(int i = 0; i < in_size; ++i) {
    for(int j = 0; j < inner_size; ++j) {
      input[i][j] = input[i][j] + loss[i];
    }
  }
  return input;
}

std::vector<double> CNN::updateWeights(std::vector<double> input,
                                       std::vector<double> loss, double lr) {
  //double prod = loss * lr;
  int in_size = input.size();
  for(int i = 0; i < in_size; ++i) {
    input[i] = input[i] + loss[i];
  }
  return input;
}

void CNN::backProp(std::vector<double> y,
                   std::vector<double> y_hat) {
  std::vector<double> loss = lossFunc_deriv(y, y_hat);
  // Update weights in fullyConnected
  wFC = updateWeights(wFC, loss, lr);
  // Hardcoded for now...
  w1 = updateWeights(w1, loss, lr);
  w2 = updateWeights(w2, loss, lr);
  w3 = updateWeights(w3, loss, lr);
  w4 = updateWeights(w4, loss, lr);
  w5 = updateWeights(w5, loss, lr);
}
