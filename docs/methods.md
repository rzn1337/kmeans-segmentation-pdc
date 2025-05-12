# K-Means Image Segmentation: Methodology

This document details the algorithm design, implementation approaches, and optimization strategies for our K-means image segmentation project.

## 1. Algorithm Overview

K-means clustering is an unsupervised machine learning algorithm that partitions a dataset into K distinct clusters based on similarity. In the context of image segmentation, we treat each pixel as a data point in feature space (RGB or other color space) and group similar pixels together.

The key intuition is that pixels with similar colors likely belong to the same object or region in the image. By reducing the number of unique colors to K, we achieve segmentation of the image.

## 2. Initialization Strategy

We implement two initialization strategies:

### 2.1 Random Initialization
- Randomly select K pixels from the image as initial centroids
- Simple but can lead to poor convergence or suboptimal results

### 2.2 K-means++ (Improved)
- Select first centroid randomly
- For remaining centroids (2 to K):
  - Calculate distance from each pixel to its nearest existing centroid
  - Choose the next centroid with probability proportional to squared distance
- Leads to better-spread initial centroids and typically faster convergence

## 3. Assignment Step

In this step, each pixel is assigned to the nearest centroid based on Euclidean distance in color space.

### Formula
For each pixel p with color values (r, g, b), we compute the distance to each centroid c_i:

```
dist(p, c_i) = sqrt((p_r - c_i_r)² + (p_g - c_i_g)² + (p_b - c_i_b)²)
```

The pixel is assigned to the centroid that gives the minimum distance:

```
cluster(p) = argmin_i(dist(p, c_i))
```

### Pseudocode
```
for each pixel p in image:
    min_dist = INFINITY
    assigned_cluster = -1
    
    for k = 0 to K-1:
        d = distance(p, centroids[k])
        if d < min_dist:
            min_dist = d
            assigned_cluster = k
            
    assignments[p] = assigned_cluster
```

## 4. Update Step

After assignment, we recalculate each centroid as the mean of all pixels assigned to it.

### Formula
For each cluster k:

```
centroid_k = (1/N_k) * Σ p_i
```

where N_k is the number of pixels assigned to cluster k, and the sum is over all such pixels.

### Pseudocode
```
for k = 0 to K-1:
    sum_r = sum_g = sum_b = 0
    count = 0
    
    for each pixel p in image:
        if assignments[p] == k:
            sum_r += p.r
            sum_g += p.g
            sum_b += p.b
            count++
            
    if count > 0:
        centroids[k].r = sum_r / count
        centroids[k].g = sum_g / count
        centroids[k].b = sum_b / count
```

## 5. Convergence Criteria

We use two convergence criteria:

1. **Maximum iterations**: Stop after a fixed number of iterations
2. **Centroid movement**: Stop when centroids move less than a threshold ε

```
is_converged = true
for k = 0 to K-1:
    if distance(old_centroids[k], new_centroids[k]) > ε:
        is_converged = false
        break
```

## 6. Parallelization Approach

### 6.1 Assignment Step Parallelization

The assignment step is embarrassingly parallel as each pixel can be processed independently:

```
#pragma omp parallel for
for each pixel p in image:
    // Assign pixel to nearest centroid
```

### 6.2 Update Step Parallelization

The update step requires more careful handling to avoid race conditions. We use two approaches:

#### 6.2.1 OpenMP Reduction
```
// Initialize per-cluster accumulators and counts
#pragma omp parallel for reduction(+:cluster_sums[:K][3], cluster_counts[:K])
for each pixel p in image:
    cluster = assignments[p]
    cluster_sums[cluster][0] += p.r
    cluster_sums[cluster][1] += p.g
    cluster_sums[cluster][2] += p.b
    cluster_counts[cluster]++
```

#### 6.2.2 Thread-Local Storage
```
// Each thread accumulates in its local storage
#pragma omp parallel
{
    // Create thread-local accumulators
    vector<vector<double>> local_sums(K, vector<double>(3, 0));
    vector<int> local_counts(K, 0);
    
    // Process subset of pixels
    #pragma omp for
    for each pixel p in image:
        cluster = assignments[p]
        local_sums[cluster][0] += p.r
        local_sums[cluster][1] += p.g
        local_sums[cluster][2] += p.b
        local_counts[cluster]++
    
    // Combine results
    #pragma omp critical
    {
        for k = 0 to K-1:
            cluster_sums[k][0] += local_sums[k][0]
            cluster_sums[k][1] += local_sums[k][1]
            cluster_sums[k][2] += local_sums[k][2]
            cluster_counts[k] += local_counts[k]
    }
}
```

## 7. Complexity Analysis

### 7.1 Sequential Version

- **Time Complexity**: O(I × K × N)
  - I = number of iterations
  - K = number of clusters
  - N = number of pixels
- **Space Complexity**: O(N + K)
  - N for pixel data and assignments
  - K for centroids

### 7.2 Parallel Version

- **Time Complexity**: O(I × K × N / P)
  - P = number of processors/threads
  - Achieves near-linear speedup for large images when P << N
- **Space Complexity**: O(N + K × P)
  - Additional space for per-thread accumulators

### 7.3 Memory Access Patterns

- Sequential version: Simple but potentially cache-inefficient
- Parallel version: Optimized to maximize cache coherence by:
  - Processing image in tiles
  - Storing pixels in structure-of-arrays format
  - Prefetching data when possible

## 8. Optimization Strategies

1. **Distance calculation optimization**:
   - Precompute squared values
   - Avoid square root until necessary
   - Use squared distance for comparisons

2. **Memory layout optimization**:
   - Align data structures to cache lines
   - Use contiguous memory for pixel data

3. **Workload balancing**:
   - Use dynamic scheduling for irregular workloads
   - Set chunk size based on cache size

4. **SIMD vectorization**:
   - Leverage vector instructions for distance calculations
   - Process multiple pixels in parallel at instruction level

5. **Early termination**:
   - Skip reassignment if pixel hasn't moved
   - Track percentage of changed assignments