#pragma once
// src/include/kmeans.h
// Common declarations for K-Means algorithm

#include <opencv2/core/core.hpp>
#include <string>
#include "../utils/logger.h"

/**
 * Run sequential K-Means clustering on an image
 * 
 * @param image Input image
 * @param k Number of clusters
 * @param maxIterations Maximum number of iterations
 * @param convergenceThreshold Threshold for convergence
 * @param iterations Output parameter for actual iterations executed
 * @param logger Logger for output messages
 * @return Segmented image
 */
cv::Mat runKMeansSequential(
    const cv::Mat& image,
    int k,
    int maxIterations,
    float convergenceThreshold,
    int& iterations,
    Logger& logger
);

/**
 * Run parallel K-Means clustering on an image
 * 
 * @param image Input image
 * @param k Number of clusters
 * @param maxIterations Maximum number of iterations
 * @param convergenceThreshold Threshold for convergence
 * @param numThreads Number of threads to use
 * @param iterations Output parameter for actual iterations executed
 * @param logger Logger for output messages
 * @return Segmented image
 */
cv::Mat runKMeansParallel(
    const cv::Mat& image,
    int k,
    int maxIterations,
    float convergenceThreshold,
    int numThreads,
    int& iterations,
    Logger& logger
);