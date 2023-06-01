
FLAGS = -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_FLAGS = -std=c++11 -I /usr/local/include/opencv4

default: outputs/hough
	./outputs/hough ./pgm_files/runway.pgm

outputs/hough: SharedMemoryHough.cu common/pgm.cpp
	nvcc $(CUDA_FLAGS) SharedMemoryHough.cu common/pgm.cpp -o outputs/hough $(FLAGS) -arch=sm_86
