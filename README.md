# ThermaState

Temperature-aware learned index with dual-agent RL control and asynchronous reorganization, integrated as a Flink StateBackend.

## Directory Structure

```
thermastate-release/
├── build-all.sh                            ← One-click build script
├── LICENSE
├── README.md
│
├── thermastate/                            ← Java/Flink StateBackend
│   ├── pom.xml
│   ├── thermastate_config.json             ← Index config (rootFanout, fanOuts)
│   ├── src/main/java/thermastate/
│       ├── ThermaStateMap.java            ← Two-step lookup: Index → ValueBundleBlock
│       ├── ValueBundleBlock.java          ← Paged flat storage (PAGE_SIZE=1024)
│       │
│       ├── agent/                         ← JNA wrappers for C++ inference
│       │   ├── ThermaStateInference.java  ← Native lib lazy-load, graceful fallback
│       │   ├── RouteAgent.java            ← Global Q-Network → root/inner fanout
│       │   ├── TreePlan.java              ← LeafAgent recursive tree planner
│       │   ├── LeafAgent.java             ← Temperature-aware leaf type decision
│       │   ├── LeafQNetwork.java          ← Leaf Q-Network inference
│       │   └── QNetwork.java              ← Base Q-Network forward pass
│       │
│       ├── config/ConfigLoader.java       ← JSON config parser
│       ├── controller/Controller.java     ← GA search + Q-Network scoring
│       │
│       ├── index/
│       │   ├── Index.java                 ← Main index: bulkLoad, get, put, reorganize
│       │   ├── Node.java                  ← Abstract node (key range + children)
│       │   ├── InnerNode.java             ← Hot routing: linear interpolation
│       │   ├── ColdInnerNode.java         ← Cold routing: B+tree separators + binary search
│       │   ├── DataNode.java              ← Hash-table data node
│       │   ├── Leaf.java                  ← Leaf base + DeltaBuffer for HotWriteLeaf
│       │   ├── LeafType.java              ← Leaf type enum {HR, HW, Cold}
│       │   ├── HotReadLeaf.java           ← Read-optimized: linear model + local scan
│       │   ├── HotWriteLeaf.java          ← Write-optimized: slack factor + overflow chain
│       │   ├── ColdLeaf.java              ← Cold storage: sorted array + binary search
│       │   ├── BPlusTree.java             ← B+tree for ColdInnerNode/ColdLeaf overflow
│       │   ├── KeyEncoder.java            ← <kg,sid,uk> → 52-bit double
│       │   └── Configuration.java         ← Index structure parameters
│       │
│       ├── monitor/
│       │   ├── TemperatureMonitor.java    ← Sliding-window temperature tracking
│       │   ├── InnerNodeStats.java        ← Per-InnerNode traffic counters
│       │   ├── LeafStats.java             ← Per-Leaf read/write counters
│       │   └── MonitorSummary.java        ← Aggregated monitoring snapshot
│       │
│       ├── reorganizer/
│       │   ├── Reorganizer.java           ← 3-stage pipeline coordinator (2s scan)
│       │   ├── ActivityTracker.java       ← Per-leaf activity score
│       │   ├── IntervalActivityTracker.java ← Per-interval-root activity (stage-2 trigger)
│       │   ├── DriftDetector.java         ← Global drift detection
│       │   ├── SubtreeRebuilder.java      ← Copy-on-Rebuild subtree reconstruction
│       │   ├── GlobalRebuilder.java       ← Full index rebuild from Route Agent
│       │   └── IntervalLock.java          ← Fine-grained interval read/write lock
│       │
│       └── state/
│           ├── ThermaStateBackend.java    ← Flink StateBackend entry point
│           ├── ThermaKeyedStateBackend.java ← KeyedStateBackend implementation
│           ├── ThermaValueState.java      ← ValueState (get/put/update)
│           └── ThermaAggregatingState.java ← AggregatingState (accumulate/merge)
│       └── src/test/java/thermastate/
│           ├── FlinkE2ETest.java           ← Full Flink pipeline test
│           ├── index/LeafCeilingTest.java   ← Single-node ceiling + force rebuild
│           └── reorganizer/
│               ├── IntervalActivityTrackerTest.java ← Stage-2 trigger scoring
│               └── SubtreeRebuildTest.java           ← Subtree rebuild + data integrity
│
└── ThermaIndex/                           ← C++ training & inference engine
    ├── CMakeLists.txt
    ├── data/model/
    │   ├── AC_Q_Net.pt                    ← Route Agent (Global Q-Network)
    │   ├── AC_Q_Scalar.pt                 ← RewardScalar mean/var
    │   ├── leaf_agent.pt                  ← LeafAgent (temperature-aware)
    │   └── tmp.pt                         ← Small Q-Network (legacy TreePlan)
    └── index/
        ├── include/
        │   ├── Parameter.h                ← Training hyperparameters
        │   ├── Index.hpp                  ← Core index (DataNode, InnerNode, bulkLoad)
        │   ├── RL_network.hpp             ← Q-Network definitions (CNN 1D + MLP)
        │   ├── TreePlanner.hpp            ← LeafAgent recursive tree constructor
        │   ├── Controller.hpp             ← GA search over fanout space
        │   ├── Configuration.hpp          ← Index structural config
        │   ├── BPlusTree.hpp              ← B+tree implementation
        │   ├── LeafTypes.hpp              ← Leaf type definitions
        │   └── experience.hpp             ← Experience replay buffer
        └── workspace/
            ├── inference_bridge.cpp        ← JNA bridge (all inference entry points)
            ├── train.cpp                   ← Global Q-Network training
            ├── train_leaf_agent.cpp        ← LeafAgent recursive DQN training
            ├── train_TSMDP.cpp             ← Small Q-Network training
            ├── train_DARE.cpp              ← DARE training
            ├── train_tree.cpp              ← Tree structure training
            ├── global_on_policy.cpp        ← Experience generation (parallel)
            ├── index_bridge.cpp            ← Index operations bridge
            ├── export_config.cpp           ← Configuration export to JSON
            ├── gen_exp.hpp                 ← Experience generation utilities
            └── train.hpp                   ← Training utilities
```

## Architecture

```
Flink Streaming Job
  │
  ThermaStateBackend  →  ThermaKeyedStateBackend
  │
  ThermaStateMap
  ├── Index (tree)
  │   ├── InnerNode (Hot)       — linear interpolation routing
  │   ├── ColdInnerNode         — B+tree separators + binary search
  │   ├── HotReadLeaf           — read-optimized: linear model + local scan
  │   ├── HotWriteLeaf          — write-optimized: slack factor + overflow chain
  │   └── ColdLeaf              — cold storage: sorted array + binary search
  └── ValueBundleBlock          — paged flat storage

  Background Adaptation (Reorganizer)
  Monitor → ActivityTracker → DriftDetector
     → SubtreeRebuilder / GlobalRebuilder

  RL Inference (JNA → C++)
  RouteAgent  →  Global_Q_network   (root/inner fanout)
  TreePlan    →  LeafAgent          (tree structure)
  LeafAgent   →  LeafAgentNetwork   (leaf type decision)
```

## Dependencies

### Java

| Dependency | Version | Scope |
|------------|---------|-------|
| JDK | 11+ | compile |
| Apache Flink | 1.17.0 | compile |
| JNA | 5.14.0 | compile |
| Deeplearning4j | 1.0.0-M2.1 | compile |
| ND4J | 1.0.0-M2.1 | compile |
| SLF4J | 1.7.36 | compile |
| Logback | 1.2.11 | runtime |
| JUnit | 4.13.2 | test |

### C++

| Dependency | Version | Notes |
|------------|---------|-------|
| GCC | 11+ | C++17 required |
| CMake | 3.22+ | |
| LibTorch | 2.0.x | CPU-only sufficient |
| Boost | 1.74+ | math, fiber, filesystem |

LibTorch CPU-only install:
```bash
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip
```

## Data

Training requires binary key-value datasets in the format `[key:double][value:double]...` with `.data` extension, placed under `ThermaIndex/data/data_set/`.

Included in the release:
- Pre-trained model weights in `ThermaIndex/data/model/`
- Synthetic Zipf datasets (α ∈ [0.5, 2.0]) generated from C++ `DataSet.hpp::random_dataset()`

## Build

### One-Click

```bash
bash build-all.sh --run-test
```

Options: `--skip-cpp`, `--skip-java`, `--run-test`

### Java

```bash
cd thermastate
mvn compile
```

### C++

```bash
cd ThermaIndex

# Download LibTorch (CPU-only)
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip

# Build
cmake -B build -DCMAKE_PREFIX_PATH=$(pwd)/libtorch
cmake --build build
```

Training executables:
```bash
./build/train              # Global Q-Network (Route Agent)
./build/train_leaf_agent   # LeafAgent recursive DQN
./build/train_TSMDP        # Small Q-Network (TreePlan)
```

### Inference Library (JNA)

The inference bridge produces the shared library used by Java:
```bash
cd ThermaIndex/build
# Linux:   libthermastate_inference.so
# Windows: thermastate_inference.dll
```

On Linux/WSL2, ensure the `.so` is on `java.library.path`:
```bash
java -Djava.library.path=ThermaIndex/build ...
```

## Usage

### Basic — Auto-Detect Models

```java
import thermastate.state.ThermaStateBackend;

ThermaStateBackend backend = new ThermaStateBackend();
```

The backend auto-scans for model files relative to the working directory.

### Explicit Configuration

```java
Configuration config = new Configuration();
config.rootFanOut = 64;
for (int i = 0; i < 256; i++) config.fanOuts[0][i] = 8;

ThermaStateBackend backend = new ThermaStateBackend(
    config,
    "ThermaIndex/data/model/AC_Q_Net.pt",
    "ThermaIndex/data/model/AC_Q_Scalar.pt",
    "ThermaIndex/data/model/leaf_agent.pt",
    1000   // bootstrap threshold
);
```

### System Property Overrides

```
-Dthermastate.global.model.path=/path/to/AC_Q_Net.pt
-Dthermastate.global.scalar.path=/path/to/AC_Q_Scalar.pt
-Dthermastate.leaf.model.path=/path/to/leaf_agent.pt
-Dthermastate.bootstrap.threshold=5000
-Dthermastate.lib.path=/path/to/libthermastate_inference.so
```

### Flink Integration

```java
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
env.setStateBackend(new ThermaStateBackend());
```

## License

Copyright (c) 2025-2026 ADA Lab, Soochow University

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
