/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

/**
 * JNA bridge — exposes C++ Hits::Index<double,double> to Java.
 *
 * Key insight: Java hashes state keys to double, uses Index as key→index mapper.
 * Actual values live in a Java-side ArrayList; Index stores the array index as "value".
 */
#include "../include/Index.hpp"
#include "../include/Configuration.hpp"

#if defined(_WIN32) || defined(_WIN64)
  #define TS_EXPORT __declspec(dllexport)
#else
  #define TS_EXPORT __attribute__((visibility("default")))
#endif

using IndexType = Hits::Index<double, double>;

extern "C" {

TS_EXPORT IndexType* thermastate_index_create(
    float rootFanOut, const float* innerFanouts, double lower, double upper)
{
    Hits::Configuration conf;
    conf.root_fan_out = rootFanOut;
    for (int i = 0; i < INNER_FANOUT_COLUMN; i++) {
        conf.fan_outs[0][i] = innerFanouts[i];
    }
    return new IndexType(conf, lower, upper);
}

TS_EXPORT void thermastate_index_destroy(IndexType* idx) {
    delete idx;
}

TS_EXPORT void thermastate_index_bulk_load(
    IndexType* idx, const double* keys, int count)
{
    std::vector<std::pair<double, double>> data;
    data.reserve(count);
    for (int i = 0; i < count; i++) {
        // value = insertion order index (0, 1, 2, ...)
        data.emplace_back(keys[i], (double)i);
    }
    std::sort(data.begin(), data.end());
    idx->bulk_load(data.begin(), data.end());
}

TS_EXPORT int thermastate_index_get(IndexType* idx, double key, double* value) {
    return idx->get(key, *value) ? 1 : 0;
}

TS_EXPORT int thermastate_index_put(IndexType* idx, double key, double value) {
    return idx->add(key, value) ? 1 : 0;
}

TS_EXPORT int thermastate_index_erase(IndexType* idx, double key) {
    return idx->erase(key) ? 1 : 0;
}

TS_EXPORT float thermastate_index_memory(IndexType* idx) {
    return idx->memory_occupied();
}

}  // extern "C"
