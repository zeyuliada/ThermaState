#!/bin/bash
#
# build-all.sh — one-click build for ThermaState
#
# Prerequisites:
#   Java: JDK 11+, Maven 3.8+
#   C++:  GCC 11+, CMake 3.22+, LibTorch 2.0.x (CPU-only), Boost 1.74+
#
# Usage:
#   bash build-all.sh [--skip-cpp] [--skip-java] [--run-test]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKIP_CPP=false
SKIP_JAVA=false
RUN_TEST=false

for arg in "$@"; do
    case "$arg" in
        --skip-cpp)  SKIP_CPP=true  ;;
        --skip-java) SKIP_JAVA=true ;;
        --run-test)  RUN_TEST=true  ;;
        *) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

echo "========================================="
echo " ThermaState — Build All"
echo "========================================="

# ── C++ ──────────────────────────────────
if [ "$SKIP_CPP" = false ]; then
    echo ""
    echo "── Building C++ (ThermaIndex) ──────────"

    cd "$SCRIPT_DIR/ThermaIndex"

    if [ ! -d "build" ]; then
        cmake -B build -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${PWD}/libtorch}"
    fi
    cmake --build build

    echo "C++ build complete."
else
    echo "Skipping C++ build."
fi

# ── Java ─────────────────────────────────
if [ "$SKIP_JAVA" = false ]; then
    echo ""
    echo "── Building Java (thermastate) ──────────"

    cd "$SCRIPT_DIR/thermastate"
    mvn compile -q

    echo "Java build complete."
else
    echo "Skipping Java build."
fi

# ── Flink E2E test ───────────────────────
if [ "$RUN_TEST" = true ]; then
    echo ""
    echo "── Running Flink E2E test ───────────────"

    cd "$SCRIPT_DIR/thermastate"
    mvn test -Dtest=thermastate.FlinkE2ETest -pl .
    echo ""
    echo "Flink E2E test PASSED."
fi

echo ""
echo "========================================="
echo " Build finished successfully."
echo "========================================="
