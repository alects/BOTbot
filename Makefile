CC = g++
flags = -O3 -g -Wall -std=c++11

image_sources = ImageProcess.cc
sources = main.cc CNN.cc NNet.cc
image_target = wrangle
target = net

dirs = $(source_dir) $(bw_dir)/2d $(edge_dir)/2d
source_dir = color_img
bw_dir = bw_img
edge_dir = edge_img

opencv = `pkg-config opencv --cflags --libs`
libraries = $(opencv)

# compiles the neural network code
$(target) : $(sources) $(dirs)
	$(CC) $(flags) -o $@ $(sources) $(libraries)

# creates directories for images
# and the vector directories within them
$(source_dir) :
	mkdir $@

$(bw_dir) :
	mkdir $@

$(bw_dir)/2d : $(bw_dir)
	mkdir -p $@

$(edge_dir) :
	mkdir $@

$(edge_dir)/2d : $(edge_dir)
	mkdir -p $@

# compiles the image processing code
wrangle : $(image_sources) $(dirs)
	$(CC) $(flags) -o $@ $(image_sources) $(libraries)

clear :
	clear && clear

clean :
	rm -rf $(image_target) $(image_target).dSYM $(target) $(target).dSYM $(bw_dir) $(edge_dir)
