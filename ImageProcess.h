#ifndef IMAGEPROCESS_H
#define IMAGEPROCESS_H
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

class ImageProcess {
public:
  const short WHITE = 255;
  const short BLACK = 0;
  // these are the maximum dimensions of the dataset(s) we're using
  // we need everything to have the same dimensions, so this is for that
  const int MAX_Y = 1090;
  const int MAX_X = 1710;
  // The stride number of pixels you move per iteration when creating the binary image.
  // The output image will have dimensions 1/STRIDE * its initial dimensions.
  const int STRIDE;
  // Sensitivity is how black a pixel needs to be to get detected
  // Higher SENS => less black
  const float SENS;

  ImageProcess() : STRIDE(1), SENS(1/3.1415926535897932) {};
  ImageProcess(int stride) : STRIDE(stride), SENS(1/3.1415926535897932) {};
  ImageProcess(int stride, float sensitivity) : STRIDE(stride), SENS(1/sensitivity) {};

  // This function will turn a greyscale cv::Mat object into a 1D
  // vector with only values of WHITE or BLACK (i.e. a binary image).
  std::vector<short> to_binary(cv::Mat in_img,
                               int rows, int cols);

  // This function will take a binary image, detect the edges in the image
  // and return a 1D vector with all the pixels.
  std::vector<short> to_edge(cv::Mat in_img,
                             int rows, int cols);
private:
  // this is the folder that contains the initial images
  const std::string COLOR_FOLDER = "color_img/";
  // the folder that contains the binary images
  const std::string BW_FOLDER = "bw_img/";
  // the folder that contains the edge detected images
  const std::string EDGE_FOLDER = "edge_img/";
};

#endif
