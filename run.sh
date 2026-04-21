#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p build output

# Configure and build (Release by default)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Try common executable names
EXE=""
for cand in optical_flow opt-flow-parallel xcorr_parallel; do
  if [[ -x "build/$cand" ]]; then
    EXE="build/$cand"
    break
  fi
done

if [[ -z "$EXE" ]]; then
  echo "Error: built executable not found in ./build."
  echo "Look for the target name in CMakeLists.txt."
  exit 1
fi

# Run with sample inputs (or with CLI arguments if provided)
usage() {
  echo "Usage: $0 <image1> <image2> [additional image paths...]"
  echo "If no arguments are given, the script will use ./data/rm1.jpg ./data/rm2.jpg."
  exit 1
}

if [[ "${1-}" == "-h" || "${1-}" == "--help" ]]; then
  usage
fi

if [[ "$#" -gt 0 ]]; then
  IMGS=("$@")
  if [[ "${#IMGS[@]}" -lt 2 ]]; then
    echo "Error: at least two image paths are required."
    usage
  fi
# else
#   IMGS=("./data/rm1.jpg" "./data/rm2.jpg")
fi

"$EXE" "${IMGS[@]}"