#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "opencv2/opencv.hpp"

const int degree_increment = 1;
const int degree_bins = 180 / degree_increment;
const int radio_bins = 100;
const float radio_increment = degree_increment * M_PI / 180;

__constant__ float d_Cos[degree_bins];
__constant__ float d_Sin[degree_bins];

void CPUHoughTran(float radio_max, float radio_scale, int x_center, int y_center, unsigned char *pic, int w, int h, int **acc) {
    *acc = new int[radio_bins * degree_bins];
    memset(*acc, 0, sizeof(int) * radio_bins * degree_bins);

    for (int i = 0; i < w; i++)
        for (int j = 0; j < h; j++) {
            int idx = j * w + i;
            if (pic[idx] > 0) {
                int x = i - x_center;
                int y = y_center - j;
                float theta = 0;
                for (int tIdx = 0; tIdx < degree_bins; tIdx++) {
                    float r = x * cos(theta) + y * sin(theta);
                    int rIdx = round((r + radio_max) / radio_scale);
                    if (rIdx >= 0 && rIdx < radio_bins) {
                        (*acc)[rIdx * degree_bins + tIdx]++;
                    }
                    theta += radio_increment;
                }
            }
        }
}

__global__ void GPUHoughTran(float radio_max, float radio_scale, int x_center, int y_center, unsigned char *pic, int w, int h, int *acc) {
    __shared__ int local_acc[degree_bins * radio_bins];
    memset(local_acc, 0, sizeof(int) * radio_bins * degree_bins);
    __syncthreads();

    int global_id = blockDim.x * blockIdx.x + threadIdx.x;
    int local_id = threadIdx.x;

    if (global_id >= w * h) return;

    int x = global_id % w - x_center;
    int y = y_center - global_id / w;

    if (pic[global_id] > 0) {
        for (int tIdx = 0; tIdx < degree_bins; tIdx++) {
            float r = x * d_Cos[tIdx] + y * d_Sin[tIdx];
            int rIdx = round((r + radio_max) / radio_scale);
            if (rIdx >= 0 && rIdx < radio_bins) {
                atomicAdd(local_acc + (rIdx * degree_bins + tIdx), 1);
            }
        }
    }

//  TODO sum of acc
    __syncthreads();
    for (int x = local_id; x < degree_bins * radio_bins; x+=blockDim.x){
        atomicAdd(&acc[x], local_acc[x]);
    }

}

double get_threshold(int* h_hough, const int degree_bins, const int radio_bins){
    // avg of weights
    double sum = std::accumulate(h_hough, h_hough + degree_bins * radio_bins, 0);
    double mean = sum / (degree_bins * radio_bins);
    // std of weights
    double sq_sum = std::inner_product(h_hough, h_hough + degree_bins * radio_bins, h_hough, 0.0);
    double stdev = std::sqrt(sq_sum / (degree_bins * radio_bins) - mean * mean);
    // El threshold = avg + 2 * desviación estándar
    // return mean + 2 * stdev;
    return 1400;
}

void draw_lines(int* h_hough, double threshold, int degree_bins, const int radio_bins, float radio_scale, float radio_max, float radio_increment, int w, int h, char **argv){
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);

    for (int i = 0; i < degree_bins * radio_bins; i++) {
        if (h_hough[i] > threshold) {
            float r = (i / degree_bins) * radio_scale - radio_max;
            float theta = (i % degree_bins) * radio_increment;
            float a = std::cos(theta), b = std::sin(theta);
            int x0 = a * r + w / 2, y0 = b * r + h / 2;
            cv::Point pt1, pt2;
            pt1.x = cvRound(x0 + 1000 * (-b));
            pt1.y = cvRound(y0 + 1000 * (a));
            pt2.x = cvRound(x0 - 1000 * (-b));
            pt2.y = cvRound(y0 - 1000 * (a));
            cv::line(img, pt1, pt2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
    }

    cv::imwrite("outputs/output.png", img);
}

int main(int argc, char **argv) {
    PGMImage inImg(argv[1]);
    int i;
    int *cpuht;
    int h = inImg.y_dim, w = inImg.x_dim;
    float milliseconds = 0;

    //  general data
    float radio_max = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
    float radio_scale = 2 * radio_max / radio_bins;
    //  calculate center
    int x_center = w / 2;
    int y_center = h / 2;

    //  CPU calculation
    CPUHoughTran(radio_max, radio_scale, x_center, y_center, inImg.pixels, w, h, &cpuht);

    //  GPU calculation
    //  pre-compute values to be stored
    float *h_Cos = (float *) malloc(sizeof(float) * degree_bins);
    float *h_Sin = (float *) malloc(sizeof(float) * degree_bins);
    float rad = 0;
    for (i = 0; i < degree_bins; i++) {
        h_Cos[i] = cos(rad);
        h_Sin[i] = sin(rad);
        rad += radio_increment;
    }
    cudaMemcpyToSymbol(d_Cos, h_Cos, sizeof(float) * degree_bins);
    cudaMemcpyToSymbol(d_Sin, h_Sin, sizeof(float) * degree_bins);

    // setup and copy data from host to device
    unsigned char *d_in, *h_in;
    int *d_hough, *h_hough;

    h_in = inImg.pixels;
    h_hough = (int *) malloc(degree_bins * radio_bins * sizeof(int));

    cudaMalloc((void **) &d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **) &d_hough, sizeof(int) * degree_bins * radio_bins);
    cudaMemcpy(d_in, h_in, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degree_bins * radio_bins);

    int blockNum = ceil(w * h / 256.0);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //  Record the start event
    cudaEventRecord(start, NULL);
    GPUHoughTran <<< blockNum, 256 >>>(radio_max, radio_scale, x_center, y_center, d_in, w, h, d_hough);
    cudaDeviceSynchronize();
    //  Record the stop event
    cudaEventRecord(stop, NULL);
    //  Wait for the stop event to complete
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    // Get results from device
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degree_bins * radio_bins, cudaMemcpyDeviceToHost);

    printf("Kernel execution time: %f mis\n", milliseconds);

    // calculate the threshold
    double threshold = get_threshold(h_hough, degree_bins, radio_bins);

    // Draw the selected lines
    draw_lines(h_hough, threshold, degree_bins, radio_bins, radio_scale, radio_max, radio_increment, w, h, argv);

    // compare CPU and GPU results
    for (i = 0; i < degree_bins * radio_bins; i++) {
        if (cpuht[i] != h_hough[i])
            printf("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
    }
    printf("Done!\n");

    // Clean-up
    cudaFree((void *) d_in);
    cudaFree((void *) d_hough);

    delete[]h_hough;
    delete[]cpuht;
    delete[]h_Cos;
    delete[]h_Sin;

    return 0;
}