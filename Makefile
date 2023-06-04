
FLAGS = -lcudart -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_FLAGS = -std=c++11 -I /usr/local/include/opencv4
PROGRAM_NAME1= GlobalMemoryHough.cu
PROGRAM_NAME2= ConstantMemoryHough.cu
PROGRAM_NAME3= SharedMemoryHough.cu

IMAGE1=runway.pgm
IMAGE2=cuadrosHough.pgm
IMAGE3=sudoku.pgm

default: outputs/hough
	./outputs/hough ./pgm_files/$(IMAGE3)

outputs/hough: common/pgm.cpp
	nvcc $(CUDA_FLAGS) $(PROGRAM_NAME2) common/pgm.cpp -o outputs/hough $(FLAGS) -arch=sm_86
