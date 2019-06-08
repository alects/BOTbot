#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "ImageProcess.h"

std::vector<short> ImageProcess::to_binary(cv::Mat in_img,
                                           int rows, int cols) {
  // loop through image and create a new black and white silhouette
  std::vector<short> out_v;
  for (int i = 0; i < rows; i+=STRIDE) {
    for (int j = 0; j < cols; j+=STRIDE) {
      // if the pixel is dark-ish set it to black
      if (in_img.at<uchar>(i,j) > WHITE/SENS) {
        out_v.push_back(BLACK);
      // else set it to white
      } else {
        out_v.push_back(WHITE);
      }
      // fill in the remaining columns
      if (j+STRIDE >= cols) {
        out_v.insert(out_v.end(), (MAX_X/STRIDE)-(j/STRIDE+1), WHITE);
      }
    }
    // fill in the remaining rows
    if (i+STRIDE >= rows) {
      out_v.insert(out_v.end(), (MAX_X/STRIDE)*(MAX_Y/STRIDE - i/STRIDE + 1), WHITE);
    }
  }

  return out_v;
}

std::vector<short> ImageProcess::to_edge(cv::Mat in_img,
                                         int rows, int cols) {
  // loop through array and create an outline
  std::vector<short> out_v;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      // sum the pixels around the target
      int sum = in_img.at<uchar>(i, j);
      int count = 0;
      // check for left border
      if (i > 0) {
        sum += in_img.at<uchar>(i-1, j);
        count++;
        // check for top border
        if (j > 0) {
          sum += in_img.at<uchar>(i-1, j-1);
          count++;
        }
        // check for bottom border
        if (j < rows) {
          sum += in_img.at<uchar>(i-1, j+1);
          count++;
        }
      }
      // check for right border
      if (i < cols) {
        sum += in_img.at<uchar>(i+1, j);
        count++;
        // check for top border
        if (j > 0) {
          sum += in_img.at<uchar>(i+1, j-1);
          count++;
        }
        // check for bottom border
        if (j < rows) {
          sum += in_img.at<uchar>(i+1, j+1);
          count++;
        }
      }
      // check top border
      if (j > 0) {
        sum += in_img.at<uchar>(i, j-1);
        count++;
      }
      // check bottom border
      if (j < rows) {
        sum += in_img.at<uchar>(i, j+1);
        count++;
      }
      // calculate average of border pixels
      int average = sum / count;
      // add a white pixel if any change was detected
      if (average < WHITE && average > BLACK) {
        out_v.push_back(WHITE);
      } else {
        out_v.push_back(BLACK);
      }
    }
  }

  return out_v;
}

int main(int argc, char** argv) {
  const std::string COLOR_FOLDER = "color_img/";
  const std::string BW_FOLDER = "bw_img/";
  const std::string EDGE_FOLDER = "edge_img/";

  ImageProcess myProcess(10);

  // get images from source folder
  std::vector<std::string> folder;
  std::vector<cv::String> folder_cv;
  // glob  into  cv::String
  cv::glob(COLOR_FOLDER + "*.jpg", folder_cv);
  // convert cv::String vector to std::string vector
  for (auto &i : folder_cv)
      folder.push_back(i);
  int photo_count = folder.size();

  // convert the color pictures to binary photos
  for (int k = 0; k < photo_count; ++k) {
    std::string leaf = folder[k];

    // Read the image file
    cv::Mat img = cv::imread(leaf, 0);

    // Check for file read failure
    if (img.empty()) {
      std::cout << "oops" << std::endl;
      std::cin.get();
      return 1;
    }

    // get dimensions of image
    cv::Size s = img.size();
    int rows = s.height;
    int cols = s.width;

    // convert the image to a binary image
    std::vector<short> binary_v = myProcess.to_binary(img, rows, cols);

    // make 2d arrays from the vectors
    rows = myProcess.MAX_Y/myProcess.STRIDE;
    cols = myProcess.MAX_X/myProcess.STRIDE;

    short arr[rows][cols];
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        arr[i][j] = binary_v[i*cols+j];
      }
    }

    // create image from array
    img = cv::Mat(rows, cols, CV_16SC1, arr);
    // get the leaf name by removing the name of the folder from the beginning
    // and the file extension from the end
    leaf = leaf.substr(COLOR_FOLDER.length(),
                       leaf.length() - COLOR_FOLDER.length() - 4);
    // write array to file
    cv::imwrite(BW_FOLDER + "/2d/" + leaf + ".bmp", img);
    // write vector to file
    cv::imwrite(BW_FOLDER + leaf + ".bmp", binary_v);
  }

  // get binary images from bw_folder
  folder.clear();
  folder_cv.clear();
  cv::glob(BW_FOLDER + "/2d/*.bmp", folder_cv);
  for (auto &i : folder_cv)
      folder.push_back(i);
  photo_count = folder.size();


  // convert the binary photos to edges
  for (int k = 0; k < photo_count; ++k) {
    std::string leaf = folder[k];

    // Read the image file
    cv::Mat img = cv::imread(leaf, 0);

    // Check for file read failure
    if (img.empty()) {
      std::cout << "oops" << std::endl;
      std::cin.get();
      return 1;
    }

    // get dimensions of image
    cv::Size s = img.size();
    int rows = s.height;
    int cols = s.width;

    // convert the image to an edge detected image
    std::vector<short> edge_v = myProcess.to_edge(img, rows, cols);

    // make 2d arrays from the vectors
    short arr[rows][cols];
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        arr[i][j] = edge_v[i*cols+j];
      }
    }

    // create image from array
    img = cv::Mat(rows, cols, CV_16SC1, arr);
    // get the leaf name by removing the name of the folder from the beginning
    // and the file extension from the end
    leaf = leaf.substr(BW_FOLDER.length() + 4,
                       leaf.length() - BW_FOLDER.length() - 4);
    // write array to file
    cv::imwrite(EDGE_FOLDER + "/2d/" + leaf, img);
    // write vector to file
    cv::imwrite(EDGE_FOLDER + leaf, edge_v);
  }

  return 0;
}