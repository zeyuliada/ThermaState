cmake_minimum_required(VERSION 3.22)
project(HITS)

set(CMAKE_CXX_STANDARD 17)

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /arch:AVX2 /utf-8")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2 /utf-8")
else()
    set(CMAKE_CXX_FLAGS_DEBUG "-march=native ${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# Enable dual-agent MARL: DARE (Route Agent) + TSMDP (Leaf Agent via Small_Q_network)
# add_compile_definitions(using_small_network)

# CPU-only training — no CUDA required
set(gpu_device torch::kCPU)
# Data directory relative to project root (used at runtime from build/)
set(data_father_path "\"../data/\"")

# LibTorch — pass via -DCMAKE_PREFIX_PATH=<libtorch_root>
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber date_time system filesystem)
find_package(Python3 REQUIRED COMPONENTS Development)

link_libraries("${TORCH_LIBRARIES}")
if(Boost_FOUND)
    link_libraries(${Boost_LIBRARIES})
    include_directories(${Boost_INCLUDE_DIRS})
endif()
include_directories(${Python3_INCLUDE_DIRS})

# NumPy for matplotlibcpp.h (global_on_policy)
include_directories(/usr/local/lib/python3.10/dist-packages/numpy/_core/include)

# Executable targets
add_executable(main main.cpp)

add_executable(global_on_policy index/workspace/global_on_policy.cpp)
target_link_libraries(global_on_policy PUBLIC ${TORCH_LIBRARIES} Python3::Python)

add_executable(train index/workspace/train.cpp)
target_link_libraries(train PUBLIC ${TORCH_LIBRARIES})

add_executable(train_TSMDP index/workspace/train_TSMDP.cpp)

add_executable(train_DARE index/workspace/train_DARE.cpp)

add_executable(dynamic_data benchmark/dynamic_data.cpp)

add_executable(dynamic_data2 benchmark/dynamic_data2.cpp)

add_executable(add_get_erase benchmark/add_get_erase.cpp)

add_executable(read_only benchmark/read_only.cpp)

# add_executable(test benchmark/test.cpp)  # requires external indexes (stx, LIPP, alex, pgm)

add_executable(opration_latency benchmark/opration_latency.cpp)

add_executable(scalability benchmark/scalability.cpp)
add_executable(Ablation benchmark/Ablation.cpp)

find_package(Threads REQUIRED)
add_executable(concurrency_test benchmark/concurrency_test.cpp)
target_link_libraries(concurrency_test PUBLIC Threads::Threads)

# Config export tool — grid search via Q-Network (Sprint 2)
add_executable(export_config index/workspace/export_config.cpp)
target_link_libraries(export_config PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(export_config PUBLIC ${Boost_LIBRARIES})
endif()

# Shared library for JNA inference (Sprint 2)
add_library(thermastate_inference SHARED
    index/workspace/inference_bridge.cpp
)
target_link_libraries(thermastate_inference PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(thermastate_inference PUBLIC ${Boost_LIBRARIES})
endif()
set_target_properties(thermastate_inference PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Shared library for JNA Index operations (replaces Java Index with native C++ HITS)
add_library(thermastate_index SHARED
    index/workspace/index_bridge.cpp
)
target_link_libraries(thermastate_index PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(thermastate_index PUBLIC ${Boost_LIBRARIES})
endif()
set_target_properties(thermastate_index PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Small_Q_network inference verification
add_executable(test_leaf_inference index/workspace/test_leaf_inference.cpp)
target_link_libraries(test_leaf_inference PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(test_leaf_inference PUBLIC ${Boost_LIBRARIES})
endif()

add_executable(test_tree_plan index/workspace/test_tree_plan.cpp)
target_link_libraries(test_tree_plan PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(test_tree_plan PUBLIC ${Boost_LIBRARIES})
endif()

# LeafAgentNetwork training (Sprint 5b)
add_executable(train_leaf_agent index/workspace/train_leaf_agent.cpp)
target_link_libraries(train_leaf_agent PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(train_leaf_agent PUBLIC ${Boost_LIBRARIES})
endif()

add_executable(test_leaf_agent index/workspace/test_leaf_agent.cpp)
target_link_libraries(test_leaf_agent PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(test_leaf_agent PUBLIC ${Boost_LIBRARIES})
endif()

# Index benchmark standalone
add_executable(index_bench benchmark/index_bench.cpp)
target_link_libraries(index_bench PUBLIC ${TORCH_LIBRARIES})
if(Boost_FOUND)
    target_link_libraries(index_bench PUBLIC ${Boost_LIBRARIES})
endif()

message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "JNA inference library: thermastate_inference → build/lib/")
message(STATUS "JNA index library: thermastate_index → build/lib/")
