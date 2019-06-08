#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#include "CNN.h"

// Accepts a 1D vector and a string
// Prints the 1D vector with each line starting with the index
// followed by a tab and then the contents of the vector at that index.
// All the output is sent to a file with the name that you pass
// into the function as to not clog up the terminal.
void printVector(std::vector<double> v, std::string fname) {
    int v_size = v.size();
    std::ofstream f;
    f.open(fname, std::ios_base::app);
    for (int i =0; i < v_size; ++i) {
      f << i << "\t" << v[i] << "\n";
    }
    f << "\n";
    f.close();
}

int main() {
  srand(0);
  const std::string SOURCE = "edge_img/*.bmp";
  // get images from source folder
  std::vector<std::string> folder;
  std::vector<cv::String> folder_cv;
  cv::glob(SOURCE, folder_cv);
  for (auto &i : folder_cv)
    folder.push_back(i);
  int photo_count = 10;//folder.size();
  CNN nn;

  // Loop through every image
  std::vector<double> v2;
  std::vector<std::vector<double>> input;
  for (int k = 0; k < photo_count; ++k) {
    // Read the image file
    std::string leaf = folder[k];
    cv::Mat img = cv::imread(leaf, 0);
    std::cout << leaf << "\n";

    // Check for file read failure
    if (img.empty()) {
      std::cout << "oops" << std::endl;
      std::cin.get();
      return 1;
    }


    // Make a vector from the cv::Mat object
    std::vector<short> v;
    // get the dimensions
    cv::Size s = img.size();
    int rows = s.height;
    int cols = s.width;

    // fill the vector
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        v.push_back(img.at<uchar>(i,j));
      }
    }

    // open and close the output file to clear it of content 
    // from previous runs
    std::ofstream file;
    file.open("o.txt");
    file.close();

    // convert the input vector to 2d
    input = nn.to2D(v);

    // run the feed forward and print the output to a file
    file.open("o.txt", std::ios_base::app);
    file << "feedForward(input):\n";
    file.close();
    v2 = nn.feedForward(input);
    printVector(v2, "o.txt");

    //testing backProp:
    file.open("o.txt", std::ios_base::app);
    file << "\nInitial targets and backProp(v2, targets):\n";
    file.close();

    std::vector<double> targets(10);
    targets[std::atoi(&leaf[9])] = 1.0;
    printVector(targets, "o.txt");

    nn.backProp(v2, targets);
    v2 = nn.feedForward(input);
    printVector(v2, "o.txt");
  }

    // check a prediction and print the output to a file
    std::ofstream file;
    v2 = nn.feedForward(input);
    
    file.open("o.txt", std::ios_base::app);
    file << "\nPrediction?:\n";
    file.close();
    printVector(v2, "o.txt");

  return 0;
}
