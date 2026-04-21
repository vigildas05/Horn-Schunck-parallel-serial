#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build output

# MPI executable name
MPI_EXE="build/opt_flow_mpi"

# Compile the MPI version
echo "Compiling MPI version..."
mpic++ -std=c++17 -O3 \
  -I./include \
  src/opt_flow_mpi.cpp \
  $(pkg-config --cflags --libs opencv4) \
  -o "$MPI_EXE"

if [[ ! -x "$MPI_EXE" ]]; then
  echo "Error: compilation failed, executable not created."
  exit 1
fi

echo "Compilation successful!"

# Usage/help
usage() {
  cat <<EOF
Usage: $0 [OPTIONS] IMAGE1 IMAGE2 ALPHA ITERATIONS [NUM_PROCS]

Compiles and runs the MPI optical flow code.

Arguments:
  IMAGE1      Path to first image
  IMAGE2      Path to second image
  ALPHA       Regularization parameter (e.g., 10.0)
  ITERATIONS  Number of iterations (e.g., 100)
  NUM_PROCS   Number of MPI processes (default: 4)

Options:
  -h, --help  Show this help message and exit

Examples:
  $0 ./data/frame1.jpg ./data/frame2.jpg 10.0 100 4
  $0 ./data/rm1.jpg ./data/rm2.jpg 15.0 200
EOF
}

# Parse options
if [[ ${#@} -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
  esac
fi

# Check arguments
if [[ ${#@} -lt 4 ]]; then
  echo "Error: insufficient arguments"
  usage
  exit 1
fi

IMG1="$1"
IMG2="$2"
ALPHA="$3"
ITERATIONS="$4"
NUM_PROCS="${5:-4}"  # Default to 4 processes

# Validate image files
if [[ ! -f "$IMG1" ]]; then
  echo "Error: image1 not found: $IMG1"
  exit 2
fi
if [[ ! -f "$IMG2" ]]; then
  echo "Error: image2 not found: $IMG2"
  exit 2
fi

echo ""
echo "Running MPI optical flow with:"
echo "  Image 1:    $IMG1"
echo "  Image 2:    $IMG2"
echo "  Alpha:      $ALPHA"
echo "  Iterations: $ITERATIONS"
echo "  Processes:  $NUM_PROCS"
echo ""

# Run with mpirun
mpirun -np "$NUM_PROCS" "$MPI_EXE" "$IMG1" "$IMG2" "$ALPHA" "$ITERATIONS"

echo ""
echo "Done! Output saved as flow_vis.png"
