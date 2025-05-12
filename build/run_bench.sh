#!/usr/bin/env bash

set -e      # Exit on error
set -o pipefail

# Default parameters
CLUSTERS=(2 3)
MAX_ITER=1000
THREADS=(2 4 8 16)
CONV=0.00001
LOG_DIR="$(pwd)/logs"
RESULT_DIR="$(pwd)/../data/results"
PROC_DIR="$(pwd)/../data/processed"
CSV_FILE="$LOG_DIR/benchmark_results.csv"

# Ensure directories exist
mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Prepare CSV header
echo "image,implementation,k,threads,real_sec" > "$CSV_FILE"

# Discover images
IMAGES=("$PROC_DIR"/*_proc.png)
if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "No processed images in $PROC_DIR" >&2
  exit 1
fi

# Colors for console output
green(){ echo -e "[0;32m$1[0m"; }
blue(){ echo -e "[0;34m$1[0m"; }

validate_images() {
    local img_count=${#IMAGES[@]}
    if [ $img_count -eq 0 ]; then
        log_error "No processed images found in $PROC_DIR"
        exit 1
    fi
    green "Found $img_count image(s) to process"
}

# Run benchmarks
echo && blue "Starting benchmarks..."
validate_images
for img in "${IMAGES[@]}"; do
  img_name=$(basename "$img")
  echo && blue "Benchmarking $img_name"

  # Sequential runs
  for k in "${CLUSTERS[@]}"; do
    green "[Seq] k=$k"
    # Capture only the real time in seconds
    real_time=$( { /usr/bin/time -f "%e" ../build/kmeans_seq \
      --input "$img" --clusters $k --max-iter $MAX_ITER \
      --convergence $CONV --output "$RESULT_DIR/${img_name%.*}_k${k}.png"; } 2>&1 >/dev/null | tr -d ' ')
    # Append to CSV
    printf "%s,seq,%d,1,%.3f
" "$img_name" "$k" "$real_time" >> "$CSV_FILE"
  done

  # Parallel runs
  for k in "${CLUSTERS[@]}"; do
    for t in "${THREADS[@]}"; do
      green "[Par] k=$k, threads=$t"
      real_time=$( { /usr/bin/time -f "%e" ../build/kmeans_par \
        --input "$img" --clusters $k --max-iter $MAX_ITER \
        --threads $t --convergence $CONV \
        --output "$RESULT_DIR/${img_name%.*}_k${k}_t${t}.png"; } 2>&1 >/dev/null | tr -d ' ')
      printf "%s,par,%d,%d,%.3f
" "$img_name" "$k" "$t" "$real_time" >> "$CSV_FILE"
    done
  done

done

blue "Done. CSV results in $CSV_FILE"