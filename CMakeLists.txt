cmake_minimum_required(VERSION 3.10)
project(kmeans_image_segmentation VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} 
                    ${CMAKE_SOURCE_DIR}/src/include
                    ${CMAKE_SOURCE_DIR}/src/utils)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Add compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

# Add executables
add_executable(kmeans_seq src/sequential/kmeans_seq.cpp)
target_link_libraries(kmeans_seq ${OpenCV_LIBS})

add_executable(kmeans_par src/parallel/kmeans_par.cpp)
target_link_libraries(kmeans_par ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

# Create directories if they don't exist
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/raw)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/processed)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data/results)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks/logs)

# Install targets
install(TARGETS kmeans_seq kmeans_par
        RUNTIME DESTINATION bin)

# Copy necessary scripts to build directory
configure_file(${CMAKE_SOURCE_DIR}/benchmarks/run_bench.sh 
               ${CMAKE_BINARY_DIR}/run_bench.sh 
               COPYONLY)

# Add testing (optional)
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmark target
add_custom_target(benchmark
  COMMAND bash ${CMAKE_SOURCE_DIR}/benchmarks/run_bench.sh
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/benchmarks
  DEPENDS kmeans_seq kmeans_par
  COMMENT "Run image segmentation benchmarks"
)