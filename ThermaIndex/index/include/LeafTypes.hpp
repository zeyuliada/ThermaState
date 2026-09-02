/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

//
// LeafTypes.hpp — Three leaf node types for ThermaState Leaf Agent training.
//
// Uses the same memory layout as DataNode (Hits::DataNode) so they can
// be stored in InnerNode slots interchangeably.
//
// Type tag: max_offset < 0 encodes leaf type (DataNode keeps max_offset >= 0)
//   -1 = ColdLeaf   (sorted array + binary search)
//   -2 = HotReadLeaf (sorted array + linear model + local scan)
//   -3 = HotWriteLeaf (sorted array + linear model + overflow chains)
//

#ifndef LEAF_TYPES_H
#define LEAF_TYPES_H

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include "BPlusTree.hpp"

// Forward declarations for Hits namespace globals
namespace Hits {
    extern unsigned long long node_memory_count;
    extern long long leaf_cost;
    extern long long leaf_max_cost;
}

// ── Type tag constants (negative values in max_offset field) ──
constexpr int LEAF_COLD_MARKER    = -1;
constexpr int LEAF_HOTREAD_MARKER = -2;
constexpr int LEAF_HOTWRITE_MARKER = -3;

constexpr int COLD_SCAN_RANGE = 0;       // binary search needs no scan range
constexpr int HOTREAD_SCAN_RANGE = 8;
constexpr int HOTREAD_SHIFT_RANGE = 16;
constexpr int HOTWRITE_SCAN_RANGE = 8;
constexpr int HOTWRITE_MAX_SHIFT = 4;

namespace Hits {

// ── ColdLeaf: B+tree-based cold leaf ─────────────────────────

template<class key_T, class value_T>
class ColdLeaf {
public:
    using self_type = ColdLeaf;
    using slot_type = std::pair<key_T, value_T>;

    double lower{0};
    double upper{0};
    int capacity{-1};
    int size{};
    int max_offset{LEAF_COLD_MARKER};   // type tag
    BPlusTree<key_T, value_T>* tree{nullptr};

    static self_type* new_segment(int capacity, double lower, double upper) {
        auto node = (self_type*)std::malloc(sizeof(self_type));
        node->capacity = capacity;
        node->lower = lower;
        node->upper = upper;
        node->size = 0;
        node->max_offset = LEAF_COLD_MARKER;
        node->tree = BPlusTree<key_T, value_T>::create(lower, upper);
        node_memory_count += sizeof(self_type);
        return node;
    }

    static void delete_segment(self_type* node) {
        BPlusTree<key_T, value_T>::destroy(node->tree);
        node_memory_count -= sizeof(self_type);
        std::free(node);
    }

    double forward(double key) const {
        return std::min(double(capacity - 1),
                        std::max(0.0, capacity * ((key - lower) / (upper - lower))));
    }

    // B+tree-based point query
    int find(key_T key) const { return tree->find(key); }

    int find_with_cost(key_T key) { return tree->find_with_cost(key); }

    // Bulk-load from sorted data (bottom-up build)
    template<class Iterator>
    void bulk_load(Iterator begin, Iterator end) {
        tree->bulk_load(begin, end);
        size = tree->size;
    }

    bool insert(key_T key, value_T value) {
        bool ok = tree->insert(key, value);
        if (ok) size = tree->size;
        return ok;
    }

    bool get(key_T key, value_T& out) const { return tree->get(key, out); }
};


// ── HotReadLeaf: sorted contiguous array + linear rank model + local scan ──

template<class key_T, class value_T>
class HotReadLeaf {
public:
    using self_type = HotReadLeaf;
    using slot_type = std::pair<key_T, value_T>;

    double lower{0};
    double upper{0};
    int capacity{-1};
    int size{};
    int max_offset{LEAF_HOTREAD_MARKER};  // type tag
    slot_type array[0];

    unsigned int* bitmap_start() {
        return (unsigned int*)(array + capacity);
    }

    static self_type* new_segment(int capacity, double lower, double upper) {
        int t = ((capacity >> 5) + 1);
        auto memory_size = sizeof(self_type) + sizeof(slot_type) * capacity + sizeof(unsigned int) * t;
        auto node = (self_type*)std::malloc(memory_size);
        node->capacity = capacity;
        node->lower = lower;
        node->upper = upper;
        node->size = 0;
        node->max_offset = LEAF_HOTREAD_MARKER;
        unsigned int* bit_map = node->bitmap_start();
        for (int i = 0; i < t; ++i) bit_map[i] = 0;
        for (int i = 0; i < capacity; ++i) { node->array[i].first = 0; }
        node_memory_count += memory_size;
        return node;
    }

    static void delete_segment(self_type* node) {
        node_memory_count -= sizeof(self_type) + sizeof(slot_type) * node->capacity +
                             sizeof(unsigned int) * ((node->capacity >> 5) + 1);
        std::free(node);
    }

    double forward(double key) const {
        return std::min(double(capacity - 1),
                        std::max(0.0, capacity * ((key - lower) / (upper - lower))));
    }

    // Linear model: predicts rank (index in [0, size-1])
    int predictRank(double key) const {
        if (size == 0) return 0;
        double kMin = array[0].first;
        double kMax = array[size - 1].first;
        if (kMax <= kMin) return 0;
        double pos = size * (key - kMin) / (kMax - kMin);
        int p = (int)pos;
        if (p < 0) p = 0;
        if (p >= size) p = size - 1;
        return p;
    }

    int find(double key) const {
        int pos = predictRank(key);
        // Scan left
        for (int i = pos, step = 0; step <= HOTREAD_SCAN_RANGE && i >= 0; step++, i--) {
            if (array[i].first == key) return i;
        }
        // Scan right
        for (int i = pos + 1, step = 1; step <= HOTREAD_SCAN_RANGE && i < size; step++, i++) {
            if (array[i].first == key) return i;
        }
        // Binary search fallback for skewed distributions
        int lo = 0, hi = size - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (array[mid].first == key) return mid;
            else if (array[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    int find_with_cost(double key) {
        int pos = predictRank(key);
        ++leaf_cost;
        // Scan left
        for (int i = pos, step = 0; step <= HOTREAD_SCAN_RANGE && i >= 0; step++, i--) {
            if (array[i].first == key) return i;
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, step + 1);
        }
        // Scan right
        for (int i = pos + 1, step = 1; step <= HOTREAD_SCAN_RANGE && i < size; step++, i++) {
            if (array[i].first == key) return i;
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, step + 1);
        }
        // Binary search fallback for skewed distributions
        int lo = 0, hi = size - 1;
        while (lo <= hi) {
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, 1);
            int mid = (lo + hi) >> 1;
            if (array[mid].first == key) return mid;
            else if (array[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    // Bulk-load insert: data arrives sorted, just append
    int find_insert(double key) {
        if (size > 0 && array[size - 1].first == key) return -1;
        if (size >= capacity) return -1;
        return size;
    }
};


// ── HotWriteLeaf: sorted contiguous array + linear model + overflow chains ──

template<class key_T, class value_T>
class HotWriteLeaf {
public:
    using self_type = HotWriteLeaf;
    using slot_type = std::pair<key_T, value_T>;

    // Overflow linked list node (stored in the overflow area after bitmap)
    struct Overflow {
        key_T key;
        value_T value;
        Overflow* next;
    };

    double lower{0};
    double upper{0};
    int capacity{-1};
    int size{};
    int max_offset{LEAF_HOTWRITE_MARKER};  // type tag
    slot_type array[0];

    unsigned int* bitmap_start() {
        return (unsigned int*)(array + capacity);
    }

    // Overflow heads: array of Overflow* pointers, stored after bitmap
    Overflow** overflow_heads() {
        int t = ((capacity >> 5) + 1);
        unsigned int* bm = bitmap_start();
        return (Overflow**)(bm + t);
    }

    // Total allocated size including overflow head pointers
    static size_t alloc_size(int capacity) {
        int t = ((capacity >> 5) + 1);
        return sizeof(self_type)
             + sizeof(slot_type) * capacity
             + sizeof(unsigned int) * t
             + sizeof(Overflow*) * capacity;
    }

    static self_type* new_segment(int capacity, double lower, double upper) {
        size_t memory_size = alloc_size(capacity);
        auto node = (self_type*)std::malloc(memory_size);
        node->capacity = capacity;
        node->lower = lower;
        node->upper = upper;
        node->size = 0;
        node->max_offset = LEAF_HOTWRITE_MARKER;
        int t = ((capacity >> 5) + 1);
        unsigned int* bit_map = node->bitmap_start();
        for (int i = 0; i < t; ++i) bit_map[i] = 0;
        for (int i = 0; i < capacity; ++i) {
            node->array[i].first = 0;
            node->overflow_heads()[i] = nullptr;
        }
        node_memory_count += memory_size;
        return node;
    }

    static void delete_segment(self_type* node) {
        // Free overflow chains
        Overflow** heads = node->overflow_heads();
        for (int i = 0; i < node->capacity; ++i) {
            Overflow* ov = heads[i];
            while (ov) {
                Overflow* next = ov->next;
                std::free(ov);
                ov = next;
            }
        }
        node_memory_count -= alloc_size(node->capacity);
        std::free(node);
    }

    double forward(double key) const {
        return std::min(double(capacity - 1),
                        std::max(0.0, capacity * ((key - lower) / (upper - lower))));
    }

    int predictRank(double key) const {
        if (size == 0) return 0;
        double kMin = array[0].first;
        double kMax = array[size - 1].first;
        if (kMax <= kMin) return 0;
        double pos = size * (key - kMin) / (kMax - kMin);
        int p = (int)pos;
        if (p < 0) p = 0;
        if (p >= size) p = size - 1;
        return p;
    }

    int find(double key) const {
        int pos = predictRank(key);
        Overflow** heads = overflow_heads();
        // Scan left
        for (int i = pos, step = 0; step <= HOTWRITE_SCAN_RANGE && i >= 0; step++, i--) {
            if (array[i].first == key) return i;
            for (Overflow* ov = heads[i]; ov; ov = ov->next) {
                if (ov->key == key) return i;
            }
        }
        // Scan right
        for (int i = pos + 1, step = 1; step <= HOTWRITE_SCAN_RANGE && i < size; step++, i++) {
            if (array[i].first == key) return i;
            for (Overflow* ov = heads[i]; ov; ov = ov->next) {
                if (ov->key == key) return i;
            }
        }
        // Binary search fallback for skewed distributions
        int lo = 0, hi = size - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (array[mid].first == key) return mid;
            // Also scan overflow chains of checked slots
            for (Overflow* ov = heads[mid]; ov; ov = ov->next) {
                if (ov->key == key) return mid;
            }
            if (array[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    int find_with_cost(double key) {
        int pos = predictRank(key);
        ++leaf_cost;
        Overflow** heads = overflow_heads();
        // Scan left
        for (int i = pos, step = 0; step <= HOTWRITE_SCAN_RANGE && i >= 0; step++, i--) {
            if (array[i].first == key) return i;
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, step + 1);
            for (Overflow* ov = heads[i]; ov; ov = ov->next) {
                ++leaf_cost;
                if (ov->key == key) return i;
            }
        }
        // Scan right
        for (int i = pos + 1, step = 1; step <= HOTWRITE_SCAN_RANGE && i < size; step++, i++) {
            if (array[i].first == key) return i;
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, step + 1);
            for (Overflow* ov = heads[i]; ov; ov = ov->next) {
                ++leaf_cost;
                if (ov->key == key) return i;
            }
        }
        // Binary search fallback for skewed distributions
        int lo = 0, hi = size - 1;
        while (lo <= hi) {
            ++leaf_cost;
            leaf_max_cost = std::max<long long>(leaf_max_cost, 1);
            int mid = (lo + hi) >> 1;
            if (array[mid].first == key) return mid;
            for (Overflow* ov = heads[mid]; ov; ov = ov->next) {
                ++leaf_cost;
                if (ov->key == key) return mid;
            }
            if (array[mid].first < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    // Bulk-load insert: data arrives sorted, just append
    int find_insert(double key) {
        if (size > 0 && array[size - 1].first == key) return -1;
        if (size >= capacity) return -1;
        return size;
    }
};


}  // namespace Hits

#endif  // LEAF_TYPES_H
