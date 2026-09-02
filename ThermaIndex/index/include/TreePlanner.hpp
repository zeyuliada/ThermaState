/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

//
// TreePlanner.hpp — Two-phase tree structure planner (DARE + TSMDP).
//
// DARE phase (upper levels): fanout from Configuration.fan_outs table
// TSMDP phase (lower levels): Small_Q_network decides expand vs leaf type
//
// Output: preorder PlanNode array consumed by Java to build the index tree.
//

#ifndef TREE_PLANNER_H
#define TREE_PLANNER_H

#include <vector>
#include <cmath>
#include <algorithm>
#include "Index.hpp"

struct PlanNode {
    int16_t action;   // leaf: 0=Cold,1=HotRead,2=HotWrite; inner: fanout (2..256)
    int8_t  is_leaf;  // 0=inner node, 1=leaf
    int8_t  reserved;

    PlanNode() : action(0), is_leaf(0), reserved(0) {}
    PlanNode(bool leaf, int16_t act) : action(act), is_leaf(leaf ? 1 : 0), reserved(0) {}
};

// Standalone get_fanout (same logic as Index::get_fanout, without the class).
inline int dare_get_fanout(const Hits::Configuration& conf, int layer,
                           double node_interval_point, double global_lower, double global_upper) {
    double pred = std::min(double(INNER_FANOUT_COLUMN - 1),
                           std::max(0.0, double(INNER_FANOUT_COLUMN - 1)
                                         * (node_interval_point - global_lower) / (global_upper - global_lower)));
    int left = int(pred);
    pred -= left;
    return int(0.5 + conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left] * (1 - pred)
                  + conf.fan_outs[std::min(INNER_FANOUT_ROW - 1, layer)][left + 1] * pred);
}

// Compute h-level boundary: upper h-1 levels = DARE, level h-1 onward = TSMDP.
inline int compute_h_level(int data_count) {
    int h = 2;
    long long capacity = 1024;
    while (capacity < data_count) {
        capacity *= 1024;
        ++h;
    }
    return h;
}

// Slice parent PDF to estimate child PDF for sub-interval.
inline std::vector<float> slice_pdf(const std::vector<float>& parent_pdf,
                                     double parent_lo, double parent_hi,
                                     double child_lo, double child_hi) {
    int n = (int)parent_pdf.size();
    std::vector<float> child(n, 0.0f);
    double parent_range = parent_hi - parent_lo;
    if (parent_range <= 0) { child[0] = 1.0f; return child; }

    int bin_start = std::max(0, std::min(n - 1, int((child_lo - parent_lo) / parent_range * n)));
    int bin_end   = std::max(0, std::min(n - 1, int((child_hi - parent_lo) / parent_range * n)));
    float mass = 0.0f;
    for (int i = bin_start; i <= bin_end; ++i) mass += parent_pdf[i];
    if (mass <= 0) {
        child[bin_start] = 1.0f;
        return child;
    }
    // Copy and re-normalise
    for (int i = 0; i < n; ++i) {
        double bin_lo = parent_lo + parent_range * i / n;
        double bin_hi = parent_lo + parent_range * (i + 1) / n;
        // Fraction of this parent bin that falls within child interval
        double overlap_lo = std::max(bin_lo, child_lo);
        double overlap_hi = std::min(bin_hi, child_hi);
        if (overlap_hi > overlap_lo) {
            child[i] = parent_pdf[i] * float((overlap_hi - overlap_lo) / (bin_hi - bin_lo)) / mass;
        }
    }
    return child;
}

// ── TSMDP phase ──────────────────────────────────────────────────

// Recursively plan subtree using LeafAgentNetwork.
// When q_network is null, defaults to Cold leaf.
inline void plan_leaf_agent(LeafAgentNetwork* q_network,
                             const std::vector<float>& pdf,
                             int data_count,
                             double lo, double hi,
                             float fr, float fw,
                             std::vector<PlanNode>& plan) {
    if (q_network == nullptr || data_count <= DATA_NODE_SIZE) {
        plan.emplace_back(true, 0);  // Cold leaf
        return;
    }

    // LeafAgentNetwork: one forward → 11-dim Q vector for all actions
    auto pdf_t  = torch::tensor(pdf).view({1, SMALL_PDF_SIZE}).to(GPU_DEVICE);
    auto size_t = torch::tensor((float)data_count).view({1, 1}).to(GPU_DEVICE);
    auto fr_t   = torch::tensor(fr).view({1, 1}).to(GPU_DEVICE);
    auto fw_t   = torch::tensor(fw).view({1, 1}).to(GPU_DEVICE);

    torch::Tensor q_values;
    {
        torch::NoGradGuard no_grad;
        q_values = q_network->forward(pdf_t, size_t, fr_t, fw_t).to(CPU_DEVICE);
    }

    // Mask invalid actions
    auto valid = get_valid_actions(data_count);
    for (int i = 0; i < 11; ++i) {
        bool allowed = false;
        for (int v : valid) if (i == v) { allowed = true; break; }
        if (!allowed) q_values[0][i] = -std::numeric_limits<float>::infinity();
    }

    int best = q_values.argmax(1).item().toInt();
    bool is_leaf;
    int value;
    decode_action(best, is_leaf, value);

    if (is_leaf) {
        plan.emplace_back(true, (int16_t)value);
        return;
    }

    // Expand: create inner node, slice PDF for children, recurse
    int fanout = value;
    plan.emplace_back(false, (int16_t)fanout);

    double range = hi - lo;
    for (int child = 0; child < fanout; ++child) {
        double child_lo = lo + range * child / fanout;
        double child_hi = lo + range * (child + 1) / fanout;
        auto child_pdf = slice_pdf(pdf, lo, hi, child_lo, child_hi);

        float child_mass = 0.0f;
        int bin_start = std::max(0, int((child_lo - lo) / range * SMALL_PDF_SIZE));
        int bin_end   = std::min(SMALL_PDF_SIZE - 1, int((child_hi - lo) / range * SMALL_PDF_SIZE));
        for (int b = bin_start; b <= bin_end; ++b) child_mass += pdf[b];
        int child_count = std::max(1, int(data_count * child_mass));

        plan_leaf_agent(q_network, child_pdf, child_count, child_lo, child_hi, fr, fw, plan);
    }
}

// ── DARE phase ───────────────────────────────────────────────────

inline void plan_dare(const Hits::Configuration& conf,
                      const std::vector<float>& pdf,
                      int data_count,
                      double lo, double hi,
                      double global_lo, double global_hi,
                      int depth, int h,
                      LeafAgentNetwork* q_network,
                      float fr, float fw,
                      std::vector<PlanNode>& plan) {
    // At depth h-1, hand off to TSMDP (LeafAgent)
    if (depth >= h - 1 || data_count <= DATA_NODE_SIZE) {
        plan_leaf_agent(q_network, pdf, data_count, lo, hi, fr, fw, plan);
        return;
    }

    double midpoint = (lo + hi) / 2.0;
    int fanout = dare_get_fanout(conf, depth, midpoint, global_lo, global_hi);
    if (fanout <= 1) {
        plan_leaf_agent(q_network, pdf, data_count, lo, hi, fr, fw, plan);
        return;
    }

    plan.emplace_back(false, (int16_t)fanout);

    double range = hi - lo;
    for (int child = 0; child < fanout; ++child) {
        double child_lo = lo + range * child / fanout;
        double child_hi = lo + range * (child + 1) / fanout;
        auto child_pdf = slice_pdf(pdf, lo, hi, child_lo, child_hi);

        float child_mass = 0.0f;
        int bin_start = std::max(0, int((child_lo - lo) / range * SMALL_PDF_SIZE));
        int bin_end   = std::min(SMALL_PDF_SIZE - 1, int((child_hi - lo) / range * SMALL_PDF_SIZE));
        for (int b = bin_start; b <= bin_end; ++b) child_mass += pdf[b];
        int child_count = std::max(1, int(data_count * child_mass));

        plan_dare(conf, child_pdf, child_count, child_lo, child_hi,
                  global_lo, global_hi, depth + 1, h, q_network, fr, fw, plan);
    }
}

// ── Top-level entry point ────────────────────────────────────────

struct BucketResult {
    int plan_offset;    // start index in the global plan array
    int plan_size;      // number of PlanNodes for this bucket
};

inline void generate_tree_plan(
    const Hits::Configuration& conf,
    LeafAgentNetwork* q_network,
    const float* pdfs,           // [num_buckets * SMALL_PDF_SIZE]
    const int* data_counts,      // [num_buckets]
    const double* lowers,        // [num_buckets]
    const double* uppers,        // [num_buckets]
    int num_buckets,
    double global_lo,
    double global_hi,
    float fr, float fw,          // temperature input (default 0.5/0.5 for bulk load)
    std::vector<PlanNode>& out_plan,
    std::vector<BucketResult>& out_buckets) {

    out_plan.clear();
    out_buckets.resize(num_buckets);

    for (int b = 0; b < num_buckets; ++b) {
        std::vector<float> pdf(pdfs + b * SMALL_PDF_SIZE,
                               pdfs + (b + 1) * SMALL_PDF_SIZE);

        int h = compute_h_level(data_counts[b]);
        int start = (int)out_plan.size();

        plan_dare(conf, pdf, data_counts[b], lowers[b], uppers[b],
                  global_lo, global_hi, 0, h, q_network, fr, fw, out_plan);

        out_buckets[b] = {start, (int)out_plan.size() - start};
    }
}

#endif  // TREE_PLANNER_H
