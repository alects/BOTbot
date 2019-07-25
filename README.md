BOTbot CNN Tree Classifier 1.0 README Updated: 6/7/19
PROGRAM AUTHORS
Brendon Kieser Irfan Filipovic Alex Petzold

BRIEF DESCRIPTION
BOTbot is a Convolutional Neural Network designed to classify trees based on leaf shape

CREATION DOCUMENTATION
BOTbot was made as a project for CIS 330 C/C++ at University Oregon during the spring term of 2019.

SETUP INSTRUCTIONS
Compiling
Download the code from our repository (linked in the Repository section below).
Within the directory that contains the Makefile, type in the following commands: $ make $ make wrangle
You should now have three directories named color_img, bw_img, and edge_img, and two other new files named wrangle and net.
Image Processing
In order to run the network, you need to have pictures of the correct dimension in the edge_img directory. To do this, you need to first run wrangle with images inside color_img.

Download the input files linked in the Repository section below. a. The input files are split up into a set of training photos and a set of testing photos.
Extract the .tar a. This should give you a folder containing all the photos in the set.
Move the photos into the color_img directory
You can also simply navigate to the color_img directory inside your terminal and enter this command: $ curl -o train_maple.tar https://ix.cs.uoregon.edu/~bkieser/files/cis-330/train_maple.tar && tar -xvf train_maple.tar && mv TrainingMaple/ . && rm -rf TrainingMaple/ train_maple.tar This will do all the above steps automatically for the training photo set, as well as remove the resulting folder and the .tar that gets downloaded. Important note: Make sure you’re in the color_img* directory because if you execute that command it’ll put 1475 files into wherever you are.

Once you have the photos inside the color_img directory, you can start the image wrangling process. If you followed the compilation steps, you should be able to run wrangle with the command $ ./wrangle and that will populate the color_img and edge_img directories. Once edge_img has photos in it, you can run the neural network.

Neural Network
Due to the complications that have been mentioned, our neural network doesn’t actually work, but if you still want to see what the output is like, then when all the steps above are complete, you can use the command $ ./net in your terminal when you’re inside the main project folder.

For further information on BOTbot, please see: Project_Report.pdf

SOFTWARE DEPENDENCIES
BOTbot was written in C++ and requires a gcc or g++ compiler. In addition, user must have openCV installed.

AUX File INFORMATION
NNet.cc and NNet.h contain a simple neural network that was used for our educational purposes and to test alongside CNN.
There is code at top of NNet.cc that can be used to test NNet. Simply replace main() in main.cc with the code and run net as usual: ./net

TrainingData.cc and TrainingData.h contains implementation for a input.txt formatted specifically. Left for future testing purposes.

Datasets
Leafsnap

One-hundred plant species leaves
