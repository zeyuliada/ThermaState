/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

/**
 * JNA inference bridge — exposes Global_Q_network + RewardScalar to Java.
 *
 * Compiled as a shared library (libthermastate_inference.so / .dll).
 * Java calls via JNA, batching all candidates into one forward pass.
 */
#include "../include/Index.hpp"
#include "../include/experience.hpp"
#include "../include/TreePlanner.hpp"
#include <boost/math/distributions/normal.hpp>

#if defined(_WIN32) || defined(_WIN64)
  #define TS_EXPORT __declspec(dllexport)
#else
  #define TS_EXPORT __attribute__((visibility("default")))
#endif

struct ThermaStateHandle {
    std::shared_ptr<Global_Q_network> q_model;
    std::shared_ptr<RewardScalar> scalar;
};

extern "C" {

TS_EXPORT ThermaStateHandle* thermastate_load(const char* model_path, const char* scalar_path) {
    auto* h = new ThermaStateHandle();
    h->q_model = std::make_shared<Global_Q_network>(Global_Q_network());
    h->q_model->eval();
    h->q_model->to(GPU_DEVICE);
    h->scalar = std::make_shared<RewardScalar>(RewardScalar(/*size=*/2));

    if (IsFileExist(model_path)) {
        torch::load(h->q_model, std::string(model_path));
        h->scalar->load(std::string(scalar_path));
        h->q_model->to(GPU_DEVICE);
    }
    return h;
}

TS_EXPORT int thermastate_predict(
    ThermaStateHandle* handle,
    const float* pdf,          // [batch * 16384]
    const float* data_size,    // [batch]
    const float* root_fanout,  // [batch]
    const float* inner_fanout, // [batch * 256]
    int batch_size,
    float* output              // [batch * 2]  interleaved (mem, get)
) {
    if (!handle || batch_size <= 0) return -1;

    auto pdf_t  = torch::from_blob((void*)pdf,          {batch_size, PDF_SIZE},
                                    torch::kFloat32).to(GPU_DEVICE);
    auto val_t  = torch::from_blob((void*)data_size,    {batch_size, 1},
                                    torch::kFloat32).to(GPU_DEVICE);
    auto root_t = torch::from_blob((void*)root_fanout,  {batch_size, 1},
                                    torch::kFloat32).to(GPU_DEVICE);
    auto inner_t= torch::from_blob((void*)inner_fanout, {batch_size, INNER_FANOUT_ROW, INNER_FANOUT_COLUMN},
                                    torch::kFloat32).to(GPU_DEVICE);

    torch::Tensor pred;
    {
        torch::NoGradGuard no_grad;
        pred = handle->q_model->forward(pdf_t, val_t, root_t, inner_t);
        pred = handle->scalar->inverse(pred).to(CPU_DEVICE);
    }

    std::memcpy(output, pred.data_ptr<float>(), batch_size * REWARD_SIZE * sizeof(float));
    return 0;
}

TS_EXPORT int thermastate_search(
    ThermaStateHandle* handle,
    const float* pdf,          // [16384]
    float data_size,
    float* out_root_fanout,    // [1]
    float* out_inner_fanout    // [256]
) {
    if (!handle) return -1;

    constexpr int MAX_GEN = 50;
    constexpr int EQ_MAX = 3;
    constexpr int MIN_ITERATION = 10;
    constexpr float MUTATION_RATE = 0.05f;
    constexpr float MUTATION_SIZE_ROOT = 128.0f;
    constexpr float MUTATION_SIZE_INNER = 3.0f;

    experience_t buffer;
    std::copy(pdf, pdf + BUCKET_SIZE, buffer.distribution);
    buffer.data_size = data_size;

    std::vector<experience_t> gens;
    int eq = 0;
    float last_reward = -std::numeric_limits<float>::max();
    experience_t best;

    for (int iter = 0; ; iter++) {
        // 1. Random initialization
        for (int i = 0; i < MAX_GEN; i++) {
            buffer.conf = Hits::Configuration::random_configuration();
            gens.push_back(buffer);
        }

        // 2. Mutation — 3 variants per candidate
        for (int i = 0; i < MAX_GEN; i++) {
            for (int v = 0; v < 3; v++) {
                experience_t mut = gens[i];
                switch (e() % 4) {
                    case 0: mut.conf.root_fan_out = shrink_root_fan_out(gens[i].conf.root_fan_out * (1+MUTATION_RATE)); break;
                    case 1: mut.conf.root_fan_out = shrink_root_fan_out(gens[i].conf.root_fan_out * (1-MUTATION_RATE)); break;
                    case 2: mut.conf.root_fan_out = shrink_root_fan_out(gens[i].conf.root_fan_out + MUTATION_SIZE_ROOT); break;
                    case 3: mut.conf.root_fan_out = shrink_root_fan_out(gens[i].conf.root_fan_out - MUTATION_SIZE_ROOT); break;
                }
                for (int j = 0; j < INNER_FANOUT_ROW; j++) {
                    for (int k = 0; k < INNER_FANOUT_COLUMN; k++) {
                        switch (e() % 4) {
                            case 0: mut.conf.fan_outs[j][k] = shrink_inner_fan_out(gens[i].conf.fan_outs[j][k] * (1+MUTATION_RATE)); break;
                            case 1: mut.conf.fan_outs[j][k] = shrink_inner_fan_out(gens[i].conf.fan_outs[j][k] * (1-MUTATION_RATE)); break;
                            case 2: mut.conf.fan_outs[j][k] = shrink_inner_fan_out(gens[i].conf.fan_outs[j][k] + MUTATION_SIZE_INNER); break;
                            case 3: mut.conf.fan_outs[j][k] = shrink_inner_fan_out(gens[i].conf.fan_outs[j][k] - MUTATION_SIZE_INNER); break;
                        }
                    }
                }
                gens.push_back(mut);
            }
        }

        // 3. Crossover — random pick from 2-4 neighbors
        for (int extract : {2, 3, 4}) {
            for (int i = 0; i < MAX_GEN - extract; i++) {
                experience_t child = gens[(e() % extract) + i];
                child.conf.root_fan_out = shrink_root_fan_out(child.conf.root_fan_out);
                for (int j = 0; j < INNER_FANOUT_ROW; j++)
                    for (int k = 0; k < INNER_FANOUT_COLUMN; k++)
                        child.conf.fan_outs[j][k] = shrink_inner_fan_out(child.conf.fan_outs[j][k]);
                gens.push_back(child);
            }
        }

        // 4. Crossover — mean with normal sampling
        for (int extract : {2, 3, 4}) {
            for (int i = 0; i < MAX_GEN - extract; i++) {
                auto mean = Hits::Configuration::zeros();
                auto var = Hits::Configuration::zeros();
                for (int ii = 0; ii < extract; ii++) {
                    mean = mean + gens[ii + i].conf;
                }
                mean = mean / float(extract);
                for (int ii = 0; ii < extract; ii++) {
                    auto tmp = gens[ii + i].conf - mean;
                    var = var + (tmp * tmp);
                }
                var = var / float(extract - 1);
                var = var.sqrt_invert();
                experience_t child = buffer;
                child.conf.root_fan_out = shrink_root_fan_out(
                    float(boost::math::quantile(boost::math::normal_distribution<>(
                        mean.root_fan_out, var.root_fan_out), 0.001 + random_u_0_1() * 0.998)));
                for (int j = 0; j < INNER_FANOUT_ROW; j++) {
                    for (int jj = 0; jj < INNER_FANOUT_COLUMN; jj++) {
                        child.conf.fan_outs[j][jj] = shrink_inner_fan_out(
                            float(boost::math::quantile(boost::math::normal_distribution<>(
                                mean.fan_outs[j][jj], var.fan_outs[j][jj]), 0.001 + random_u_0_1() * 0.998)));
                    }
                }
                gens.push_back(child);
            }
        }

        // 5. Batch evaluate via Q-Network
        auto tensors = ExpGenerator::experience_to_tensor(gens);
        auto pdf_t  = std::get<0>(tensors).to(GPU_DEVICE);
        auto val_t  = std::get<1>(tensors).to(GPU_DEVICE);
        auto root_t = std::get<2>(tensors).to(GPU_DEVICE);
        auto inner_t= std::get<3>(tensors).to(GPU_DEVICE);

        torch::Tensor pred;
        {
            torch::NoGradGuard no_grad;
            pred = handle->q_model->forward(pdf_t, val_t, root_t, inner_t);
            pred = handle->scalar->inverse(pred).to(CPU_DEVICE);
        }

        auto reward_arr = pred.data_ptr<float>();
        for (auto& gen : gens) {
            gen.cost.memory = reward_arr[0];
            gen.cost.get = reward_arr[1];
            reward_arr += REWARD_SIZE;
        }

        // 6. Sort by reward descending
        std::sort(gens.begin(), gens.end(), [](experience_t& a, experience_t& b) {
            float ra = -(a.cost.memory * MEMORY_WEIGHT + a.cost.get * QUERY_WEIGHT);
            float rb = -(b.cost.memory * MEMORY_WEIGHT + b.cost.get * QUERY_WEIGHT);
            return ra > rb;
        });

        // 7. Convergence check
        float current_reward = -(gens[0].cost.memory * MEMORY_WEIGHT + gens[0].cost.get * QUERY_WEIGHT);
        if (current_reward == last_reward) {
            eq++;
            if (eq >= EQ_MAX && iter > MIN_ITERATION) {
                best = gens[0];
                break;
            }
        } else {
            last_reward = current_reward;
            eq = 0;
        }
        best = gens[0];

        // 8. Truncate
        gens.resize(MAX_GEN);
    }

    *out_root_fanout = best.conf.root_fan_out;
    std::copy((float*)best.conf.fan_outs, (float*)best.conf.fan_outs + INNER_FANOUT_SIZE, out_inner_fanout);

    return 0;
}

TS_EXPORT void thermastate_free(ThermaStateHandle* handle) {
    delete handle;
}

// ── Small_Q_network (TSMDP / Leaf Agent) ──────────────────────────────

struct LeafHandle {
    std::shared_ptr<Small_Q_network> model;
    std::vector<int> action_space = {0,1,2,3,4,5,6,7,8,9,10};  // 11 Leaf Agent actions
};

TS_EXPORT LeafHandle* thermastate_load_leaf(const char* model_path) {
    auto* h = new LeafHandle();
    h->model = std::make_shared<Small_Q_network>(Small_Q_network());
    h->model->eval();
    h->model->to(GPU_DEVICE);

    if (IsFileExist(model_path)) {
        torch::load(h->model, std::string(model_path));
        h->model->to(GPU_DEVICE);
    }
    return h;
}

TS_EXPORT int thermastate_leaf_decide(
    LeafHandle* handle,
    const float* pdf,       // [SMALL_PDF_SIZE=1024]
    float data_count,
    float* out_result       // [2]: out[0]=is_leaf(0/1), out[1]=value(leaf_type or fanout)
) {
    if (!handle) return -1;

    int n_actions = (int)handle->action_space.size();

    // Batch PDF: repeat across all actions
    auto pdf_t = torch::from_blob((void*)pdf, {1, SMALL_PDF_SIZE},
                                   torch::kFloat32)
                     .mul(torch::ones({n_actions, SMALL_PDF_SIZE}))
                     .to(GPU_DEVICE);

    // Data count: repeat across actions
    auto val_t = torch::tensor(data_count).view({1, 1})
                     .mul(torch::ones({n_actions, 1}))
                     .to(GPU_DEVICE);

    // Action IDs
    auto act_t = torch::tensor(handle->action_space)
                     .to(torch::kFloat32).view({n_actions, 1})
                     .to(GPU_DEVICE);

    torch::Tensor pred;
    {
        torch::NoGradGuard no_grad;
        pred = handle->model->forward(pdf_t, val_t, act_t).to(CPU_DEVICE);
    }

    int best_ac = torch::argmax(pred, 0).item().toInt();
    bool is_leaf;
    int value;
    decode_action(best_ac, is_leaf, value);
    out_result[0] = is_leaf ? 1.0f : 0.0f;
    out_result[1] = (float) value;
    return 0;
}

TS_EXPORT void thermastate_free_leaf(LeafHandle* handle) {
    delete handle;
}

// ── LeafAgentNetwork (temperature-aware leaf type decision) ──────

struct LeafAgentHandle {
    std::shared_ptr<LeafAgentNetwork> model;
};

TS_EXPORT LeafAgentHandle* thermastate_leaf_agent_load(const char* model_path) {
    auto* h = new LeafAgentHandle();
    h->model = std::make_shared<LeafAgentNetwork>(LeafAgentNetwork());
    h->model->eval();
    h->model->to(GPU_DEVICE);
    if (IsFileExist(model_path)) {
        try {
            torch::load(h->model, std::string(model_path));
            h->model->to(GPU_DEVICE);
        } catch (const std::exception& e) {
            std::cerr << "[LeafAgent] load failed: " << e.what() << std::endl;
            delete h;
            return nullptr;
        }
    }
    return h;
}

/**
 * Decide the best leaf action given temperature data.
 *
 * @param handle       native handle from thermastate_leaf_agent_load
 * @param pdf          float[SMALL_PDF_SIZE=1024]  local data distribution
 * @param data_count   number of keys in this region
 * @param fr           read frequency  [0,1]
 * @param fw           write frequency [0,1]
 * @param out_result   float[3]: out[0]=is_leaf, out[1]=value(leafType|fanout), out[2]=Q[best]
 * @return 0 on success, -1 on error
 */
TS_EXPORT int thermastate_leaf_agent_decide(
    LeafAgentHandle* handle,
    const float* pdf,
    float data_count,
    float fr,
    float fw,
    float* out_result) {

    if (!handle) return -1;

    auto pdf_t  = torch::from_blob((void*)pdf, {1, SMALL_PDF_SIZE},
                                    torch::kFloat32).to(GPU_DEVICE);
    auto size_t = torch::tensor(data_count).view({1, 1}).to(GPU_DEVICE);
    auto fr_t   = torch::tensor(fr).view({1, 1}).to(GPU_DEVICE);
    auto fw_t   = torch::tensor(fw).view({1, 1}).to(GPU_DEVICE);

    torch::Tensor q_values;
    {
        torch::NoGradGuard no_grad;
        q_values = handle->model->forward(pdf_t, size_t, fr_t, fw_t).to(CPU_DEVICE);
    }

    // Mask invalid actions for small datasets
    int valid_count = (int)action_space.size();
    for (int i = 0; i < valid_count; ++i) {
        bool allowed = false;
        auto vv = get_valid_actions((int)data_count);
        for (int va : vv) {
            if ((int)action_space[i] == va) { allowed = true; break; }
        }
        if (!allowed) q_values[0][i] = -std::numeric_limits<float>::infinity();
    }

    int best_i = torch::argmax(q_values, 0).item().toInt();
    int best_action = (int)action_space[best_i];
    bool is_leaf;
    int value;
    decode_action(best_action, is_leaf, value);
    out_result[0] = is_leaf ? 1.0f : 0.0f;
    out_result[1] = (float)value;
    out_result[2] = q_values[0][best_i].item().toFloat();
    return 0;
}

TS_EXPORT void thermastate_leaf_agent_free(LeafAgentHandle* handle) {
    delete handle;
}

// ── Tree Plan (LeafAgent-powered recursive tree builder) ──────────

struct TreePlanHandle {
    std::shared_ptr<LeafAgentNetwork> leaf_agent;
    Hits::Configuration conf;
    double global_lo;
    double global_hi;
    float fr;       // read frequency for tree planning
    float fw;       // write frequency for tree planning
};

/**
 * Create a tree-plan handle.
 *
 * @param model_path    path to leaf_agent.pt, or "" for heuristic-only
 * @param fanout_table  float[INNER_FANOUT_SIZE] DARE inner fanout table, or NULL for defaults
 * @param root_fanout   DARE root fanout
 * @param global_lo     global key space lower bound
 * @param global_hi     global key space upper bound
 * @param fr            read frequency (0.5 = default for bulk load)
 * @param fw            write frequency (0.5 = default for bulk load)
 */
TS_EXPORT TreePlanHandle* thermastate_plan_create(
    const char* model_path,
    const float* fanout_table,
    float root_fanout,
    double global_lo,
    double global_hi,
    float fr, float fw) {

    auto* h = new TreePlanHandle();
    h->global_lo = global_lo;
    h->global_hi = global_hi;
    h->fr = fr;
    h->fw = fw;

    // Load LeafAgentNetwork (leaf_agent.pt)
    if (model_path && model_path[0] != '\0') {
        h->leaf_agent = std::make_shared<LeafAgentNetwork>(LeafAgentNetwork());
        try {
            torch::load(h->leaf_agent, std::string(model_path));
            h->leaf_agent->to(GPU_DEVICE);
            h->leaf_agent->eval();
        } catch (const std::exception& e) {
            std::cerr << "[TreePlan] LeafAgent load failed: " << e.what() << std::endl;
            h->leaf_agent = nullptr;
        }
    }

    // Set configuration
    h->conf.root_fan_out = root_fanout;
    if (fanout_table != nullptr) {
        std::copy(fanout_table, fanout_table + INNER_FANOUT_SIZE,
                  (float*)h->conf.fan_outs);
    } else {
        for (int j = 0; j < INNER_FANOUT_COLUMN; ++j) h->conf.fan_outs[0][j] = 8.0f;
    }

    return h;
}

/**
 * Generate the full tree plan for all root buckets using LeafAgentNetwork.
 *
 * Output packed as int32 array: 8 nodes per int, 4 bits each (action 0-10).
 * is_leaf is NOT stored — derived from action (<=2 → leaf).
 * 0xF (15) = sentinel for unused trailing slots.
 * out_offsets[] gets {plan_offset, plan_size} per bucket (node-indexed, not packed-indexed).
 *
 * @return actual number of plan nodes written, or -1 on error
 */
TS_EXPORT int thermastate_plan_generate(
    TreePlanHandle* handle,
    const float* pdfs,            // [num_buckets * SMALL_PDF_SIZE]
    const int*   data_counts,     // [num_buckets]
    const double* lowers,         // [num_buckets]
    const double* uppers,         // [num_buckets]
    int num_buckets,
    int* out_plan,                // pre-allocated, ceil(max_plan_nodes/8) int32s
    int max_plan_nodes,           // max number of nodes (NOT packed ints)
    int* out_offsets              // [num_buckets * 2]: {plan_offset, plan_size} per bucket
) {
    if (!handle || num_buckets <= 0) return -1;

    std::vector<PlanNode> plan;
    std::vector<BucketResult> buckets;

    generate_tree_plan(
        handle->conf,
        handle->leaf_agent.get(),
        pdfs, data_counts, lowers, uppers, num_buckets,
        handle->global_lo, handle->global_hi,
        handle->fr, handle->fw,
        plan, buckets);

    int total = (int)plan.size();
    if (total > max_plan_nodes) {
        std::cerr << "[TreePlan] plan too large: " << total
                  << " > " << max_plan_nodes << std::endl;
        return -1;
    }

    // Pack: 8 nodes per int32, 4 bits each (action 0-10). 0xF = sentinel.
    int nPacked = (total + 7) / 8;
    for (int i = 0; i < nPacked; ++i) {
        int packed = 0;
        for (int j = 0; j < 8; ++j) {
            int idx = i * 8 + j;
            int val = (idx < total) ? (plan[idx].action & 0xF) : 0xF;
            packed |= (val << (j * 4));
        }
        out_plan[i] = packed;
    }
    for (int b = 0; b < num_buckets; ++b) {
        out_offsets[b * 2]     = buckets[b].plan_offset;
        out_offsets[b * 2 + 1] = buckets[b].plan_size;
    }

    return total;
}

TS_EXPORT void thermastate_plan_free(TreePlanHandle* handle) {
    delete handle;
}

}  // extern "C"
