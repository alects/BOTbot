// Very basic CNN, with forward prop. Need to implement backProp and getResult
// Clean up functions and improve header file

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
class CNN {
public:
	// Max picture size is currently 171 x 109
	// Convolution layer of size 10 x 10, pool size of 2 x 2
	const int maxRow = 171;
	const int maxCol = 109;
	const int filterSize = 10;
	const int pool = 2;
	const int outputSize = 10l;
	const double lr = .1;

	std::vector<std::vector<double>> w1;
	std::vector<std::vector<double>> w2;
	std::vector<std::vector<double>> w3;
	std::vector<std::vector<double>> w4;
	std::vector<std::vector<double>> w5;
	std::vector<double> wFC;


	CNN() {
		w1 = createRandWeights();
		w2 = createRandWeights();
		w3 = createRandWeights();
		w4 = createRandWeights();
		w5 = createRandWeights();
		wFC = createFCWeights();
	};
	CNN(std::vector<std::vector<double>> w1,
      std::vector<std::vector<double>> w2,
      std::vector<std::vector<double>> w3,
      std::vector<std::vector<double>> w4,
      std::vector<std::vector<double>> w5,
      std::vector<double> wFC
      ) : w1(w1), w2(w2), w3(w3), w4(w4), w5(w5), wFC(wFC) {};

	// Convert single dimension input picture, to 2D std::vector
	std::vector<std::vector<double>> to2D(std::vector<short> in);

	double dotLeakyReLU(int i, int j,
                      std::vector<std::vector<double>> in,
			                std::vector<std::vector<double>> w);

	double poolRegion(int i, int j, std::vector<std::vector<double>> in);

	// Creates the a pooled layer, using pool function
	std::vector<std::vector<double>> poolLayer(std::vector<std::vector<double>> in,
                                             int x, int y);

	std::vector<std::vector< double>> convLayer(std::vector<std::vector<double>> in,
                                              std::vector<std::vector<double>> w, int x, int y);

	// Convolution and Pool functions intended to intialize weights and pass through 2 convs then pool
	std::vector<std::vector<double>> convPool(std::vector<std::vector<double>> in,
                                            std::vector<std::vector<double>> w1,
                                            std::vector<std::vector<double>> w2,
                                            int y, int x);

	// Implement 3 conv, and 1 pool with passed weights
	std::vector<std::vector<double>> convPool(std::vector<std::vector<double>> in,
                                            std::vector<std::vector<double>> w1,
                                            int y, int x);

	// Function to fullyConenct 2D std::vector, using 1D std::vector of weights
	std::vector<double> fullyConnect(std::vector<std::vector<double>> in,
                                   std::vector<double> w,
                                   int x, int y);

	// Taken from repo code from Alex
	static double randomWeight() { return rand() / double(RAND_MAX); }

	// Return random 2D weight std::vector
	std::vector<std::vector<double>> createRandWeights();

	// Create random weight std::vector for fully connected layer
	std::vector<double> createFCWeights();

	// Print weight vals from 2D std::vector
	void printWeight(std::vector<std::vector<double>> in,
                   int x, int y);

// FEEDFORWARD WILL HAVE 3 Pool layers, and 7 Conv layers
// Pass through 2 Conv and pool, then 2 conv and pool, then 3 conv and pool
// Then fully connect, Fully connect will have input of size 9 x 7, filter of 1 x 10

////////////////// CURRENTLY FILTER DEPTH OF 1 to reduce amount of weight std::vectors /////////////

// FeedForward first pass, create random weights then pass through
	std::vector<double> feedForward(std::vector<std::vector<double>> input);

	// FeedForward, but passed with std::vectors of weights, implement if need. above but pass with weights
	std::vector<double> lossFunc_deriv(std::vector<double> y,
                        std::vector<double> y_hat);

	std::vector<std::vector<double>> updateWeights(std::vector<std::vector<double>> input,
                                                 std::vector<double> loss, double lr);

	std::vector<double> updateWeights(std::vector<double> input,
                                    std::vector<double> loss, double lr);

	void backProp(std::vector<double> y,
                std::vector<double> y_hat);
};
