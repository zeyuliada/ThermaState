/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

/**
 * Train LeafAgentNetwork with recursive tree building and real cost measurement.
 *
 * For each sample, ALL 11 actions are evaluated:
 *   - Leaf actions (0-2): build leaf → run skewed queries → measure cost
 *   - Expand actions (3-10): build InnerNode → for each child, use the DQN
 *     itself (epsilon-greedy) to decide the child's action → recurse →
 *     sum total subtree cost
 *
 * This makes the Q-values self-consistent: an expand action's cost includes
 * whatever subtree the DQN would actually build underneath it.
 *
 * DQN: LeafAgentNetwork
 *   Input:  PDF(1024) + data_size(1) + fr(1) + fw(1) = 1027 dims
 *   Output: 11-dim Q vector → action = argmax(Q)
 *   Architecture: CNN 1D (PDF) + MLP head
 */
#include <queue>
#include <string>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include "../include/Parameter.h"
#include "../../include/DataSet.hpp"
#include "../include/Index.hpp"
#include "../include/RL_network.hpp"

#undef  BATCH_SIZE
#define BATCH_SIZE 64
#define MAX_TREES 100
#define SAMPLES_PER_TREE 30

#define LEAF_QUERY_OPS 2000
#define MAX_RECURSIVE_DEPTH 8

static float my_memory_weight() { return 0.5f; }

// ── Synthetic fr/fw ─────────────────────────────────────────────

struct SynthWL { float fr; float fw; };

static SynthWL gen_fr_fw(int idx) {
    switch (idx % 6) {
        case 0: return {float(0.90 + random_u_0_1() * 0.10), float(random_u_0_1() * 0.02)};
        case 1: return {float(0.85 + random_u_0_1() * 0.15), float(random_u_0_1() * 0.05)};
        case 2: return {float(random_u_0_1() * 0.05), float(0.85 + random_u_0_1() * 0.15)};
        case 3: return {float(random_u_0_1() * 0.02), float(0.90 + random_u_0_1() * 0.10)};
        case 4: return {float(0.35 + random_u_0_1() * 0.3), float(0.35 + random_u_0_1() * 0.3)};
        default:return {float(random_u_0_1() * 0.005), float(random_u_0_1() * 0.005)};
    }
}

// ── Skewed query generation ──────────────────────────────────────

static std::vector<int> gen_queries(int n, int data_size, float skew) {
    std::vector<int> idx(n);
    if (skew < 0.05f) {
        for (int i = 0; i < n; ++i) idx[i] = e() % data_size;
    } else {
        double alpha = std::max(0.3, (double)skew);
        for (int i = 0; i < n; ++i) {
            int j = (int)(data_size * std::pow(random_u_0_1(), 1.0 / alpha));
            if (j < 0) j = 0;
            if (j >= data_size) j = data_size - 1;
            idx[i] = j;
        }
    }
    return idx;
}

// ── Leaf action measurement ──────────────────────────────────────

template<typename KT, typename VT>
float measure_leaf_action(int action, const std::vector<std::pair<KT,VT>>& data,
                          double lo, double hi, float fr, float fw) {
    auto last_mem = Hits::node_memory_count;
    int ds = (int)data.size();
    int cap = std::max(DATA_NODE_SIZE, int(float(ds) / default_density));
    float qskew = std::max(fr, fw);

    if (action == 0) {
        auto leaf = Hits::ColdLeaf<KT,VT>::new_segment(cap, lo, hi);
        leaf->bulk_load(data.begin(), data.end());
        Hits::leaf_cost = 0;
        int n = (int)(LEAF_QUERY_OPS * std::max(fr, 0.1f));
        auto qi = gen_queries(n, ds, qskew);
        for (int i = 0; i < n; ++i)
            leaf->find_with_cost(data[qi[i]].first);
        float qc = float(Hits::leaf_cost) / float(std::max(1, n));
        float mem = float(Hits::node_memory_count - last_mem)
                    / sizeof(typename Hits::DataNode<KT,VT>::slot_type)
                    / float(std::max(1, ds));
        Hits::ColdLeaf<KT,VT>::delete_segment(leaf);
        return my_memory_weight() * mem + leaf_cost_weight * qc;
    }

    if (action == 1) {
        auto leaf = Hits::HotReadLeaf<KT,VT>::new_segment(cap, lo, hi);
        for (auto& kv : data) {
            int pos = leaf->find_insert(kv.first);
            leaf->array[pos] = kv;
            ++leaf->size;
        }
        Hits::leaf_cost = 0;
        int n = (int)(LEAF_QUERY_OPS * std::max(fr, 0.1f));
        auto qi = gen_queries(n, ds, qskew);
        for (int i = 0; i < n; ++i)
            leaf->find_with_cost(data[qi[i]].first);
        float qc = float(Hits::leaf_cost) / float(std::max(1, n));
        float mem = float(Hits::node_memory_count - last_mem)
                    / sizeof(typename Hits::DataNode<KT,VT>::slot_type)
                    / float(std::max(1, ds));
        Hits::HotReadLeaf<KT,VT>::delete_segment(leaf);
        return my_memory_weight() * mem + leaf_cost_weight * qc;
    }

    if (action == 2) {
        auto leaf = Hits::HotWriteLeaf<KT,VT>::new_segment(cap, lo, hi);
        for (auto& kv : data) {
            int pos = leaf->find_insert(kv.first);
            leaf->array[pos] = kv;
            ++leaf->size;
        }
        Hits::leaf_cost = 0;
        int n_read  = (int)(LEAF_QUERY_OPS * std::max(fr, 0.1f));
        int n_write = (int)(LEAF_QUERY_OPS * std::max(fw, 0.05f));
        auto qi = gen_queries(n_read, ds, qskew);
        for (int i = 0; i < n_read; ++i)
            leaf->find_with_cost(data[qi[i]].first);
        for (int i = 0; i < n_write; ++i) {
            auto& kv = data[i % ds];
            int pos = leaf->find_insert(kv.first + 1e-6);
            if (pos >= 0) {
                leaf->array[pos] = {kv.first, kv.second};
                ++leaf->size;
                ++Hits::leaf_cost;
            }
        }
        float qc = float(Hits::leaf_cost) / float(std::max(1, n_read + n_write));
        float mem = float(Hits::node_memory_count - last_mem)
                    / sizeof(typename Hits::DataNode<KT,VT>::slot_type)
                    / float(std::max(1, ds));
        Hits::HotWriteLeaf<KT,VT>::delete_segment(leaf);
        return my_memory_weight() * mem + leaf_cost_weight * qc;
    }

    return 1e9f;
}

// ── PDF slice (same as TreePlanner.hpp) ──────────────────────────

static std::vector<float> slice_pdf(const std::vector<float>& parent_pdf,
                                     double parent_lo, double parent_hi,
                                     double child_lo, double child_hi) {
    int n = (int)parent_pdf.size();
    std::vector<float> child(n, 0.0f);
    double parent_range = parent_hi - parent_lo;
    if (parent_range <= 0) { child[0] = 1.0f; return child; }
    float mass = 0.0f;
    for (int i = 0; i < n; ++i) {
        double bin_lo = parent_lo + parent_range * i / n;
        double bin_hi = parent_lo + parent_range * (i + 1) / n;
        double overlap_lo = std::max(bin_lo, child_lo);
        double overlap_hi = std::min(bin_hi, child_hi);
        if (overlap_hi > overlap_lo) {
            child[i] = parent_pdf[i] * float((overlap_hi - overlap_lo) / (bin_hi - bin_lo));
            mass += child[i];
        }
    }
    if (mass > 0) {
        for (int i = 0; i < n; ++i) child[i] /= mass;
    }
    return child;
}

// ── Forward-declare ──────────────────────────────────────────────

template<typename KT, typename VT>
static float measure_subtree_recursive(
    LeafAgentNetwork& model,
    const std::vector<std::pair<KT,VT>>& data,
    const std::vector<float>& pdf,
    double lo, double hi,
    float fr, float fw,
    float epsilon,
    int depth);

// ── DQN action picker (epsilon-greedy) ───────────────────────────

template<typename KT, typename VT>
static int pick_dqn_action(LeafAgentNetwork& model,
                            const std::vector<std::pair<KT,VT>>& data,
                            const std::vector<float>& pdf,
                            float fr, float fw, float epsilon) {
    int count = (int)data.size();
    auto valid = get_valid_actions(count);

    if (random_u_0_1() < epsilon) {
        return valid[e() % valid.size()];
    }

    auto pdf_t  = torch::from_blob((void*)pdf.data(), {1, SMALL_PDF_SIZE},
                                    torch::kFloat32).to(GPU_DEVICE);
    auto size_t = torch::tensor((float)count).view({1, 1}).to(GPU_DEVICE);
    auto fr_t   = torch::tensor(fr).view({1, 1}).to(GPU_DEVICE);
    auto fw_t   = torch::tensor(fw).view({1, 1}).to(GPU_DEVICE);

    torch::Tensor q;
    {
        torch::NoGradGuard no_grad;
        model.eval();  // single-sample BatchNorm requires eval mode
        q = model.forward(pdf_t, size_t, fr_t, fw_t).to(CPU_DEVICE);
    }

    for (int i = 0; i < 11; ++i) {
        bool ok = false;
        for (int v : valid) if (i == v) { ok = true; break; }
        if (!ok) q[0][i] = -std::numeric_limits<float>::infinity();
    }

    return q.argmax(1).item().toInt();
}

// ── Measure cost of a SPECIFIC expand action ─────────────────────
// Builds InnerNode(fanout), then for each child calls
// measure_subtree_recursive (DQN decides children's actions).

template<typename KT, typename VT>
static float measure_expand_recursive(
    LeafAgentNetwork& model,
    int fanout,
    const std::vector<std::pair<KT,VT>>& data,
    const std::vector<float>& pdf,
    double lo, double hi,
    float fr, float fw,
    float epsilon,
    int depth) {

    if (depth >= MAX_RECURSIVE_DEPTH || (int)data.size() <= DATA_NODE_SIZE) {
        return measure_leaf_action<KT,VT>(0, data, lo, hi, fr, fw);
    }

    auto last_mem = Hits::node_memory_count;
    auto inner = Hits::InnerNode<KT,VT>::new_segment(fanout, lo, hi);

    auto splits = Hits::split_dataset<KT,VT>(data.begin(), data.end(), inner);
    float total_child_cost = 0.0f;
    int total_child_keys = 0;

    double range = hi - lo;
    for (int ci = 0; ci < fanout && ci < (int)splits.size(); ++ci) {
        auto& s = splits[ci];
        int cs = s.first.first, ce = s.first.second;
        if (ce <= cs) continue;

        std::vector<std::pair<KT,VT>> cd(data.begin() + cs, data.begin() + ce);
        int child_ds = (int)cd.size();
        double clo = s.second.first, chi = s.second.second;

        auto child_pdf = slice_pdf(pdf, lo, hi, clo, chi);
        float child_cost = measure_subtree_recursive<KT,VT>(
            model, cd, child_pdf, clo, chi, fr, fw, epsilon, depth + 1);

        total_child_cost += child_cost * child_ds;
        total_child_keys += child_ds;
    }

    int ds = (int)data.size();
    float inner_mem = float(Hits::node_memory_count - last_mem)
                      / sizeof(typename Hits::DataNode<KT,VT>::slot_type)
                      / float(std::max(1, ds));
    float avg_child = total_child_keys > 0 ? total_child_cost / float(total_child_keys) : 0.0f;

    Hits::InnerNode<KT,VT>::delete_segment(inner);
    return my_memory_weight() * inner_mem + inner_cost_weight + avg_child;
}

// ── Recursive subtree: DQN picks action → delegate to leaf/expand ─

template<typename KT, typename VT>
static float measure_subtree_recursive(
    LeafAgentNetwork& model,
    const std::vector<std::pair<KT,VT>>& data,
    const std::vector<float>& pdf,
    double lo, double hi,
    float fr, float fw,
    float epsilon,
    int depth) {

    int ds = (int)data.size();
    if (ds == 0) return 0.0f;
    if (depth >= MAX_RECURSIVE_DEPTH || ds <= DATA_NODE_SIZE) {
        return measure_leaf_action<KT,VT>(0, data, lo, hi, fr, fw);
    }

    int action = pick_dqn_action<KT,VT>(model, data, pdf, fr, fw, epsilon);
    bool is_leaf; int value;
    decode_action(action, is_leaf, value);

    if (is_leaf) {
        return measure_leaf_action<KT,VT>(action, data, lo, hi, fr, fw);
    }
    return measure_expand_recursive<KT,VT>(
        model, value, data, pdf, lo, hi, fr, fw, epsilon, depth);
}

// ── Evaluate ALL 11 actions for one sample ───────────────────────

template<typename KT, typename VT>
static void evaluate_all_actions(
    LeafAgentNetwork& model,
    const std::vector<std::pair<KT,VT>>& data,
    const std::vector<float>& pdf,
    double lo, double hi,
    float fr, float fw,
    float epsilon,
    float* q_target)   // out: 11 floats
{
    int ds = (int)data.size();
    auto valid = get_valid_actions(ds);

    // Initialize all to -inf (invalid)
    for (int i = 0; i < 11; ++i) q_target[i] = -1e9f;

    for (int ai : valid) {
        bool is_leaf; int val;
        decode_action(ai, is_leaf, val);

        float cost;
        if (is_leaf) {
            cost = measure_leaf_action<KT,VT>(ai, data, lo, hi, fr, fw);
        } else {
            // Expand action: build InnerNode(val), then DQN decides each child recursively
            cost = measure_expand_recursive<KT,VT>(
                model, val, data, pdf, lo, hi, fr, fw, epsilon, 0);
        }
        q_target[ai] = -cost;    // Q = negative cost (higher is better)
    }
}

// ── Sample + batch training ─────────────────────────────────────

struct Sample {
    float pdf[SMALL_PDF_SIZE];
    float data_size;
    float fr;
    float fw;
    float q_target[11];
};

static void train_batch(LeafAgentNetwork& net, torch::optim::Adam& opt,
                        const std::vector<Sample>& batch) {
    int bs = (int)batch.size();
    auto pdf_t  = torch::zeros({bs, SMALL_PDF_SIZE});
    auto size_t = torch::zeros({bs, 1});
    auto fr_t   = torch::zeros({bs, 1});
    auto fw_t   = torch::zeros({bs, 1});
    auto tgt_t  = torch::zeros({bs, 11});

    auto pp = pdf_t.data_ptr<float>(), sp = size_t.data_ptr<float>();
    auto fp = fr_t.data_ptr<float>(), wp = fw_t.data_ptr<float>();
    auto tp = tgt_t.data_ptr<float>();

    for (auto& s : batch) {
        std::copy(s.pdf, s.pdf + SMALL_PDF_SIZE, pp); pp += SMALL_PDF_SIZE;
        *sp++ = s.data_size; *fp++ = s.fr; *wp++ = s.fw;
        std::copy(s.q_target, s.q_target + 11, tp); tp += 11;
    }

    pdf_t = pdf_t.to(GPU_DEVICE); size_t = size_t.to(GPU_DEVICE);
    fr_t  = fr_t.to(GPU_DEVICE);  fw_t   = fw_t.to(GPU_DEVICE);
    tgt_t = tgt_t.to(GPU_DEVICE);

    for (int epoch = 0; epoch < 50; ++epoch) {
        opt.zero_grad();
        auto loss = torch::mse_loss(net.forward(pdf_t, size_t, fr_t, fw_t), tgt_t);
        loss.backward();
        opt.step();
        static int step = 0;
        if (++step % 10 == 0)
            std::cout << "  [train] loss=" << loss.item<float>() << " step=" << step << std::endl;
    }
}

// ── Main ────────────────────────────────────────────────────────

static std::string ckpt_path()   { return model_father_path + "leaf_agent.pt"; }
static std::string optim_path()  { return model_father_path + "leaf_agent_optim.pt"; }

int main() {
    std::cout.setf(std::ios::unitbuf);
    torch::set_num_threads(1);

    auto net = std::make_shared<LeafAgentNetwork>();
    net->to(GPU_DEVICE);
    net->train();
    auto opt = torch::optim::Adam(net->parameters(), 0.001);

    int start_tree = 0;
    int total_samples = 0;
    float epsilon = 0.3f;   // exploration rate: 30% random actions during rollout

    if (IsFileExist(ckpt_path().c_str())) {
        torch::load(net, ckpt_path());
        net->to(GPU_DEVICE);
        if (IsFileExist(optim_path().c_str())) {
            torch::load(opt, optim_path());
        }
        epsilon = 0.15f;  // lower exploration on resume
        std::cout << "[Resume] Loaded leaf_agent.pt + optimizer, epsilon=" << epsilon << std::endl;
    }

    auto names = scanFiles(data_father_path);
    struct DS { std::string name; int fd; size_t sz; };
    std::vector<DS> dss;
    for (auto& nm : names) {
        std::string fp = data_father_path + nm;
        int fd = open(fp.c_str(), O_RDONLY);
        if (fd < 0) continue;
        size_t s = lseek(fd, 0, SEEK_END) / sizeof(std::pair<KEY_TYPE,VALUE_TYPE>);
        if (s >= 10000) dss.push_back({nm, fd, s}); else close(fd);
    }
    if (dss.empty()) { std::cerr << "No datasets." << std::endl; return 1; }
    std::cout << "[Data] " << dss.size() << " datasets." << std::endl;

    std::vector<Sample> buf;
    buf.reserve(BATCH_SIZE);

    for (int tid = start_tree; tid < MAX_TREES; ++tid) {
        auto& ds = dss[tid % dss.size()];
        int len = std::max(10000, std::min(50000, (int)(ds.sz * random_u_0_1_skew(0.3f))));
        int start = (int)(e() % (ds.sz - len));

        using KV = std::pair<KEY_TYPE,VALUE_TYPE>;
        std::vector<KV> data(len);
        pread(ds.fd, data.data(), len * sizeof(KV), start * sizeof(KV));
        std::sort(data.begin(), data.end(),
                  [](auto& a, auto& b) { return a.first < b.first; });
        auto ue = std::unique(data.begin(), data.end(),
                              [](auto& a, auto& b) { return a.first == b.first; });
        data.erase(ue, data.end());
        if ((int)data.size() < 2000) continue;

        for (int si = 0; si < SAMPLES_PER_TREE; ++si) {
            int sub_s = e() % std::max(1, (int)data.size() - 500);
            int sub_l = std::max(2000, std::min((int)data.size() - sub_s, 20000));
            if (sub_s + sub_l > (int)data.size()) continue;

            std::vector<KV> sub(data.begin() + sub_s, data.begin() + sub_s + sub_l);
            if ((int)sub.size() < 100) continue;
            double lo = sub[0].first, hi = sub.back().first;

            auto pdf = get_pdf<KEY_TYPE,VALUE_TYPE>(sub.begin(), sub.end(), lo, hi, SMALL_PDF_SIZE);

            Sample sp;
            std::copy(pdf.begin(), pdf.end(), sp.pdf);
            sp.data_size = (float)sub.size();
            auto wl = gen_fr_fw(total_samples);
            sp.fr = wl.fr; sp.fw = wl.fw;

            // Evaluate ALL 11 actions.
            // Expand actions use the DQN recursively to measure true cost.
            // eval() mode needed for single-sample Batchnorm in rollout.
            net->eval();
            evaluate_all_actions<KEY_TYPE,VALUE_TYPE>(
                *net, sub, pdf, lo, hi, sp.fr, sp.fw, epsilon, sp.q_target);

            buf.push_back(sp);
            ++total_samples;
            if ((int)buf.size() >= BATCH_SIZE) {
                net->train();
                train_batch(*net, opt, buf);
                buf.clear();
            }
        }

        // Decay epsilon after each tree
        epsilon = std::max(0.05f, epsilon * 0.98f);

        if (tid % 5 == 0 && tid > start_tree) {
            torch::save(net, ckpt_path());
            torch::save(opt, optim_path());
            std::cout << "[Save] tree=" << tid << " samples=" << total_samples
                      << " epsilon=" << epsilon << std::endl;
        }
    }

    if (!buf.empty()) { net->train(); train_batch(*net, opt, buf); }
    torch::save(net, ckpt_path());
    torch::save(opt, optim_path());
    std::cout << "[Done] total_samples=" << total_samples << " epsilon=" << epsilon << std::endl;
    return 0;
}
