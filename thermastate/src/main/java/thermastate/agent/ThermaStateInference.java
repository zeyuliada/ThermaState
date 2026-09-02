/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * JNA mapping to the native thermastate_inference shared library.
 *
 * The native library wraps Global_Q_network + RewardScalar + LeafAgent,
 * compiled from ThermaIndex/index/workspace/inference_bridge.cpp.
 *
 * <p>Loading is lazy and optional — when the native library is not available
 * (e.g. on Windows without a compiled DLL), {@link #isAvailable()} returns
 * false and callers should fall back to heuristic tree construction.
 */
public interface ThermaStateInference extends Library {

    /** Lazy singleton — null when the native library cannot be loaded. */
    static ThermaStateInference instance() {
        return Holder.INSTANCE;
    }

    /** True when the native inference library was loaded successfully. */
    static boolean isAvailable() {
        return Holder.INSTANCE != null;
    }

    /** Holder class defers loading until first access. */
    final class Holder {
        static final ThermaStateInference INSTANCE = load();

        private static ThermaStateInference load() {
            try {
                String path = System.getProperty("thermastate.lib.path");
                if (path != null && !path.isEmpty()) {
                    return Native.load(path, ThermaStateInference.class);
                }
                return Native.load("thermastate_inference", ThermaStateInference.class);
            } catch (UnsatisfiedLinkError e) {
                System.err.println("[ThermaState] native inference library not available: "
                    + e.getMessage());
                System.err.println("[ThermaState] Agent-driven tree planning disabled; "
                    + "will use heuristic fallback.");
                return null;
            }
        }
    }

    /** Load model and scalar from files. Returns opaque handle; null on failure. */
    Pointer thermastate_load(String modelPath, String scalarPath);

    /**
     * Batch inference — evaluates all candidates in one forward pass.
     *
     * @param handle       opaque native handle
     * @param pdf          packed float[batch * 16384]
     * @param dataSize     float[batch]  data_size per candidate
     * @param rootFanout   float[batch]  root_fanout per candidate
     * @param innerFanout  packed float[batch * 256]
     * @param batchSize    number of candidates
     * @param output       float[batch * 2] interleaved (mem, get) per candidate
     * @return 0 on success, -1 on error
     */
    int thermastate_predict(Pointer handle,
                            float[] pdf,
                            float[] dataSize,
                            float[] rootFanout,
                            float[] innerFanout,
                            int batchSize,
                            float[] output);

    /**
     * Run GA search for the optimal Configuration using the native Q-Network.
     *
     * @param handle         opaque native handle
     * @param pdf            float[16384] data distribution (PDF)
     * @param dataSize       dataset size in records
     * @param outRootFanout  float[1]  receives best root_fanout
     * @param outInnerFanout float[256] receives best inner fanout values
     * @return 0 on success, -1 on error
     */
    int thermastate_search(Pointer handle,
                           float[] pdf,
                           float dataSize,
                           float[] outRootFanout,
                           float[] outInnerFanout);

    /** Release native resources. */
    void thermastate_free(Pointer handle);

    // ── Small_Q_network (TSMDP / Leaf Agent) ──────────────────────

    /** Load Small_Q_network from tmp.pt. Returns opaque handle; null on failure. */
    Pointer thermastate_load_leaf(String modelPath);

    /**
     * Decide best action for a leaf region using 12-action Leaf Agent.
     *
     * @param handle    opaque native handle
     * @param pdf       float[1024]  local PDF of the subtree
     * @param dataCount number of keys in the subtree
     * @param outResult float[2]  out[0]=isLeaf(1/0), out[1]=leafType(0..2) or fanout(2..512)
     * @return 0 on success, -1 on error
     */
    int thermastate_leaf_decide(Pointer handle,
                                 float[] pdf,
                                 float dataCount,
                                 float[] outResult);

    /** Release leaf model native resources. */
    void thermastate_free_leaf(Pointer handle);

    // ── Leaf Agent (temperature-aware leaf type decision) ─────────

    /** Load LeafAgentNetwork from leaf_agent.pt. */
    Pointer thermastate_leaf_agent_load(String modelPath);

    /**
     * Decide best leaf action given temperature state.
     * @param handle     native handle
     * @param pdf        float[1024] local data distribution
     * @param dataCount  number of keys in region
     * @param fr         read frequency [0,1]
     * @param fw         write frequency [0,1]
     * @param outResult  float[3]: [0]=isLeaf(1/0), [1]=value, [2]=Q_value
     */
    int thermastate_leaf_agent_decide(Pointer handle,
                                       float[] pdf,
                                       float dataCount,
                                       float fr,
                                       float fw,
                                       float[] outResult);

    void thermastate_leaf_agent_free(Pointer handle);

    // ── Tree Plan (LeafAgent-powered recursive planner) ───────────

    /** Create tree-planning handle. modelPath="" for heuristic-only. */
    Pointer thermastate_plan_create(String modelPath,
                                     float[] fanoutTable,  // [256] or null
                                     float rootFanout,
                                     double globalLo,
                                     double globalHi,
                                     float fr,
                                     float fw);

    /**
     * Generate complete tree plan.
     *
     * Packed format: 8 nodes per int32, 4bit action each (LSB first).
     * Action 0-2=leaf, 3-10=expand, 0xF=sentinel. isLeaf derived from action <= 2.
     * outOffsets are node-indexed (not packed-indexed).
     *
     * @return actual node count, or -1 on error
     */
    int thermastate_plan_generate(Pointer handle,
                                   float[] pdfs,
                                   int[] dataCounts,
                                   double[] lowers,
                                   double[] uppers,
                                   int numBuckets,
                                   int[] outPlan,       // ceil(maxPlanNodes/8) int32s
                                   int maxPlanNodes,
                                   int[] outOffsets);    // [numBuckets * 2]

    void thermastate_plan_free(Pointer handle);
}
