// src/sequential/kmeans_seq.cpp
// Sequential implementation of K-Means for image segmentation

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <string>
#include <random>
#include <limits>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "../include/kmeans.h"
#include "../utils/logger.h"
#include "../utils/cli_parser.h"

// Represents a pixel as a feature vector (RGB)
struct Pixel {
    std::vector<float> features;  // RGB values normalized to [0,1]
    int cluster;                  // Assigned cluster ID
    
    Pixel(const std::vector<float>& f) : features(f), cluster(-1) {}
};

// Represents a centroid
struct Centroid {
    std::vector<float> features;  // RGB values
    int count;                    // Number of pixels assigned to this centroid
    
    Centroid(const std::vector<float>& f) : features(f), count(0) {}
    
    void reset() {
        std::fill(features.begin(), features.end(), 0.0f);
        count = 0;
    }
    
    void update(const std::vector<float>& sum) {
        if (count > 0) {
            for (size_t i = 0; i < features.size(); ++i) {
                features[i] = sum[i] / count;
            }
        }
    }
};

// Calculate Euclidean distance between two feature vectors
float euclideanDistance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float sum = 0.0f;
    for (size_t i = 0; i < v1.size(); ++i) {
        float diff = v1[i] - v2[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// Main K-Means clustering function
cv::Mat runKMeansSequential(
    const cv::Mat& image,
    int k,
    int maxIterations,
    float convergenceThreshold,
    int& iterations,
    Logger& logger
) {
    logger.log(LogLevel::INFO, "Starting sequential K-Means with k=" + std::to_string(k) + ", maxIterations=" + std::to_string(maxIterations));
    
    // Start timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Convert image to feature vectors
    int width = image.cols;
    int height = image.rows;
    int channels = image.channels();
    
    logger.log(LogLevel::INFO, "Image dimensions: " + std::to_string(width) + "x" + std::to_string(height) + " with " + std::to_string(channels) + " channels");
    
    std::vector<Pixel> pixels;
    pixels.reserve(width * height);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<float> features(channels);
            for (int c = 0; c < channels; ++c) {
                features[c] = image.at<cv::Vec3b>(y, x)[c] / 255.0f;  // Normalize to [0,1]
            }
            pixels.emplace_back(features);
        }
    }
    
    logger.log(LogLevel::INFO, "Converted image to " + std::to_string(pixels.size()) + " pixels");
    
    // Initialize centroids using k-means++ algorithm
    std::vector<Centroid> centroids;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, pixels.size() - 1);
    
    // Choose first centroid randomly
    int firstCentroidIdx = dis(gen);
    centroids.emplace_back(pixels[firstCentroidIdx].features);
    
    // Choose remaining centroids using k-means++ approach
    for (int i = 1; i < k; ++i) {
        std::vector<float> distances(pixels.size(), std::numeric_limits<float>::max());
        float totalDistance = 0.0f;
        
        // Calculate minimum distance to existing centroids
        for (size_t j = 0; j < pixels.size(); ++j) {
            for (const auto& centroid : centroids) {
                float d = euclideanDistance(pixels[j].features, centroid.features);
                distances[j] = std::min(distances[j], d);
            }
            totalDistance += distances[j] * distances[j];  // Square the distance for weighted selection
        }
        
        // Choose next centroid with probability proportional to distance
        std::uniform_real_distribution<float> distDis(0, totalDistance);
        float threshold = distDis(gen);
        float cumulativeDistance = 0.0f;
        size_t nextCentroidIdx = 0;
        
        for (size_t j = 0; j < pixels.size(); ++j) {
            cumulativeDistance += distances[j] * distances[j];
            if (cumulativeDistance >= threshold) {
                nextCentroidIdx = j;
                break;
            }
        }
        
        centroids.emplace_back(pixels[nextCentroidIdx].features);
    }
    
    logger.log(LogLevel::INFO, "Initialized " + std::to_string(centroids.size()) + " centroids");
    
    // Prepare for iterations
    bool converged = false;
    iterations = 0;
    
    // Main K-Means loop
    while (!converged && iterations < maxIterations) {
        iterations++;
        
        // Assignment step
        for (auto& pixel : pixels) {
            float minDistance = std::numeric_limits<float>::max();
            int bestCluster = -1;
            
            for (size_t c = 0; c < centroids.size(); ++c) {
                float distance = euclideanDistance(pixel.features, centroids[c].features);
                if (distance < minDistance) {
                    minDistance = distance;
                    bestCluster = c;
                }
            }
            
            pixel.cluster = bestCluster;
        }
        
        // Prepare for update step
        std::vector<std::vector<float>> sums(k, std::vector<float>(channels, 0.0f));
        std::vector<int> counts(k, 0);
        
        // Accumulate sums for each cluster
        for (const auto& pixel : pixels) {
            int cluster = pixel.cluster;
            counts[cluster]++;
            
            for (size_t f = 0; f < pixel.features.size(); ++f) {
                sums[cluster][f] += pixel.features[f];
            }
        }
        
        // Update centroids and check convergence
        float maxCentroidShift = 0.0f;
        
        for (size_t c = 0; c < centroids.size(); ++c) {
            if (counts[c] > 0) {
                std::vector<float> oldFeatures = centroids[c].features;
                
                // Update centroid
                for (size_t f = 0; f < centroids[c].features.size(); ++f) {
                    centroids[c].features[f] = sums[c][f] / counts[c];
                }
                
                // Calculate how much the centroid moved
                float shift = euclideanDistance(oldFeatures, centroids[c].features);
                maxCentroidShift = std::max(maxCentroidShift, shift);
            }
        }
        
        // Check convergence
        converged = maxCentroidShift < convergenceThreshold;
        
        logger.log(LogLevel::INFO, "Iteration " + std::to_string(iterations) + 
                  ": max centroid shift = " + std::to_string(maxCentroidShift) +
                  (converged ? " (converged)" : ""));
    }
    
    // Create segmented image
    cv::Mat segmented = cv::Mat::zeros(height, width, CV_8UC3);
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int cluster = pixels[idx].cluster;
            
            // Map cluster to color (use the centroid color)
            // for (int c = 0; c < channels; ++c) {
            //     segmented.at<cv::Vec3b>(y, x)[c] = static_cast<uchar>(centroids[cluster].features[c] * 255.0f);
            // }

            
            static const std::vector<cv::Vec3b> CLUSTER_COLORS = {
                {0,   0, 255},   // Red
                {255, 0,   255}, // Purple
                {0,   255, 0},   // Green
                {0,   255, 255}, // Yellow
                {255, 255, 0},   // Cyan
                {0,   165, 255}, // Orange
                {255, 0,   0},   // Blue
                {147, 20,  255}, // Magenta
                // add more if you need >8 clusters
            };

            // Map cluster to a fixed palette color
            const cv::Vec3b& color = CLUSTER_COLORS[cluster % CLUSTER_COLORS.size()];
            segmented.at<cv::Vec3b>(y, x) = color;


                    }
                }
    
    // End timing
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    
    logger.log(LogLevel::INFO, "K-Means completed in " + std::to_string(iterations) + 
               " iterations, total runtime: " + std::to_string(duration.count()) + " ms");
    
    return segmented;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    CLIParser parser;
    parser.addOption("--input", "Input image path", true);
    parser.addOption("--clusters", "Number of clusters (K)", true);
    parser.addOption("--max-iter", "Maximum iterations", true);
    parser.addOption("--output", "Output image path", true);
    parser.addOption("--convergence", "Convergence threshold", false, "0.001");
    parser.addOption("--log", "Log file path", false, "");
    
    if (!parser.parse(argc, argv)) {
        std::cout << "Usage: " << argv[0] << " --input <path> --clusters <k> --max-iter <n> --output <path> [--convergence <threshold>] [--log <path>]" << std::endl;
        return 1;
    }
    
    std::string inputPath = parser.getValue("--input");
    std::string outputPath = parser.getValue("--output");
    int k = std::stoi(parser.getValue("--clusters"));
    int maxIterations = std::stoi(parser.getValue("--max-iter"));
    float convergenceThreshold = std::stof(parser.getValue("--convergence"));
    std::string logPath = parser.getValue("--log");
    
    // Set up logger
    Logger logger(logPath);
    
    // Load input image
    cv::Mat inputImage = cv::imread(inputPath);
    if (inputImage.empty()) {
        logger.log(LogLevel::ERROR, "Failed to load image: " + inputPath);
        return 1;
    }
    
    logger.log(LogLevel::INFO, "Loaded input image: " + inputPath);
    
    // Run K-Means
    int iterations = 0;
    cv::Mat segmentedImage = runKMeansSequential(inputImage, k, maxIterations, convergenceThreshold, iterations, logger);
    
    // Save output image
    bool saved = cv::imwrite(outputPath, segmentedImage);
    if (!saved) {
        logger.log(LogLevel::ERROR, "Failed to save output image: " + outputPath);
        return 1;
    }
    
    logger.log(LogLevel::INFO, "Saved segmented image to: " + outputPath);
    
    // Output summary
    std::cout << "K-Means segmentation complete:" << std::endl;
    std::cout << "  Input: " << inputPath << std::endl;
    std::cout << "  Output: " << outputPath << std::endl;
    std::cout << "  Clusters (K): " << k << std::endl;
    std::cout << "  Iterations: " << iterations << " (max: " << maxIterations << ")" << std::endl;
    std::cout << std::endl;
    
    return 0;
}