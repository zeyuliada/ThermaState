/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Pointer;

import thermastate.index.LeafType;

/**
 * Temperature-aware Leaf Agent (paper Leaf Agent).
 *
 * Input:  PDF(1024) + dataSize + fr + fw  (frequency from TemperatureMonitor)
 * Output: leaf type or expand decision
 *
 * Usage:
 *   LeafAgent agent = LeafAgent.load("/path/to/leaf_agent.pt");
 *   LeafAgent.Decision d = agent.decide(pdf, dataSize, fr, fw);
 *   agent.close();
 */
public final class LeafAgent implements AutoCloseable {

    public static final int SMALL_PDF_SIZE = 1024;

    public static final class Decision {
        public final boolean isLeaf;
        public final LeafType leafType;   // valid when isLeaf
        public final int fanout;          // valid when !isLeaf
        public final float qValue;        // Q-value of best action

        public Decision(boolean isLeaf, int value, float qValue) {
            this.isLeaf = isLeaf;
            this.qValue = qValue;
            if (isLeaf) {
                this.leafType = value == 0 ? LeafType.COLD
                              : value == 1 ? LeafType.HOT_READ
                              : LeafType.HOT_WRITE;
                this.fanout = 0;
            } else {
                this.leafType = null;
                this.fanout = value;
            }
        }

        @Override
        public String toString() {
            return isLeaf
                ? String.format("Leaf(%s q=%.4f)", leafType, qValue)
                : String.format("Expand(fanout=%d q=%.4f)", fanout, qValue);
        }
    }

    private final ThermaStateInference lib;
    private final Pointer handle;

    private LeafAgent(ThermaStateInference lib, Pointer handle) {
        this.lib = lib;
        this.handle = handle;
    }

    /** Load LeafAgentNetwork from leaf_agent.pt. */
    public static LeafAgent load(String modelPath) {
        ThermaStateInference lib = ThermaStateInference.instance();
        if (lib == null) throw new RuntimeException("Native inference library not available");
        Pointer handle = lib.thermastate_leaf_agent_load(modelPath);
        if (handle == null) {
            throw new RuntimeException("Failed to load leaf agent: " + modelPath);
        }
        return new LeafAgent(lib, handle);
    }

    /**
     * Decide the best action for a leaf region.
     *
     * @param pdf       float[SMALL_PDF_SIZE] local data distribution
     * @param dataCount number of keys in this region
     * @param fr        read frequency  [0,1]
     * @param fw        write frequency [0,1]
     * @return decision with action and Q-value
     */
    public Decision decide(float[] pdf, int dataCount, float fr, float fw) {
        if (pdf.length != SMALL_PDF_SIZE) {
            throw new IllegalArgumentException(
                "pdf length must be " + SMALL_PDF_SIZE);
        }
        float[] out = new float[3];
        int rc = lib.thermastate_leaf_agent_decide(handle, pdf, (float) dataCount, fr, fw, out);
        if (rc != 0) {
            throw new RuntimeException("LeafAgent inference failed, rc=" + rc);
        }
        boolean isLeaf = out[0] != 0.0f;
        int value = (int) (out[1] + 0.5f);
        return new Decision(isLeaf, value, out[2]);
    }

    @Override
    public void close() {
        if (handle != null) {
            lib.thermastate_leaf_agent_free(handle);
        }
    }
}
