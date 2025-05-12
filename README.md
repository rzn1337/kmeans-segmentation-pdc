# K-Means Image Segmentation Project

This project implements K-means clustering for image segmentation in both sequential and parallel versions.

## Overview

The K-means algorithm is used to segment images by clustering pixels based on their color similarities. This implementation provides both sequential (single-threaded) and parallel (multi-threaded using OpenMP) versions for performance comparison.

## Features

- Sequential and parallel K-means implementations
- Configurable number of clusters (K)
- Customizable convergence parameters
- Performance benchmarking tools
- Visualization of segmentation results

## Requirements

- C++17 compiler (GCC 7+ or equivalent)
- CMake 3.10+
- OpenMP
- stb_image (included in repo)
- Python 3.6+ with matplotlib (for benchmarking visualizations)

## Building the Project

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Usage

### Sequential Version

```bash
./sequential/kmeans_seq \
  --input ../data/processed/*_proc.png \
  --clusters 8 \
  --max-iter 100 \
  --output ../data/results/image1_k8_iter100.png
```

### Parallel Version

```bash
./parallel/kmeans_par \
  --input ../data/processed/*_proc.png \
  --clusters 8 \
  --max-iter 100 \
  --threads 8 \
  --output ../data/results/image1_k8_iter100.png
```

## Project Structure

```
project-root/
├── data/
│   ├── raw/                # Original input images
│   ├── processed/          # Preprocessed images (resized, normalized)
│   └── results/            # Segmented output images
├── src/
│   ├── sequential/         # Sequential implementation
│   │   └── kmeans_seq.cpp
│   ├── parallel/           # Parallel implementation
│   │   └── kmeans_par.cpp
│   ├── include/            # Header files
│   └── utils/              # Common utilities (I/O, logging)
├── benchmarks/             # Benchmark scripts & raw logs
│   ├── run_bench.sh
│   └── logs/
├── docs/                   # Documents, reports, diagrams
│   └── methods.md
├── CMakeLists.txt          # Build configuration
└── README.md               # Project overview and instructions
```

## Benchmarking

Run the benchmark script to evaluate performance across different parameters:

```bash
cd benchmarks
./run_bench.sh
```

Results will be saved in `benchmarks/logs/` with visualizations in `benchmarks/plots/`.

## License

MIT License