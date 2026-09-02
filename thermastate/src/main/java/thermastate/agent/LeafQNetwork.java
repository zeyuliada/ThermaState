/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Pointer;

import thermastate.index.LeafType;

/**
 * Java wrapper around the native Small_Q_network (11-action Leaf Agent).
 *
 * Action space (11 total):
 *   0=Cold leaf, 1=HotRead leaf, 2=HotWrite leaf,
 *   3..10=expand with fanout = 2^(action-2)
 *
 * Usage:
 *   LeafQNetwork lqn = LeafQNetwork.load("/path/to/tmp.pt");
 *   LeafQNetwork.Decision d = lqn.decide(pdf, dataCount);
 *   // d.isLeaf → use d.leafType; else → use d.fanout
 *   lqn.close();
 */
public final class LeafQNetwork implements AutoCloseable {

    public static final int SMALL_PDF_SIZE = 1024;

    /** Result of a Leaf Agent decision. */
    public static final class Decision {
        public final boolean isLeaf;
        public final LeafType leafType;   // valid when isLeaf
        public final int fanout;          // valid when !isLeaf

        Decision(boolean isLeaf, int value) {
            this.isLeaf = isLeaf;
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
            return isLeaf ? ("Leaf(" + leafType + ")") : ("Expand(fanout=" + fanout + ")");
        }
    }

    private final ThermaStateInference lib;
    private final Pointer handle;

    private LeafQNetwork(ThermaStateInference lib, Pointer handle) {
        this.lib = lib;
        this.handle = handle;
    }

    /** Load Small_Q_network from tmp.pt file path. */
    public static LeafQNetwork load(String modelPath) {
        ThermaStateInference lib = ThermaStateInference.instance();
        if (lib == null) throw new RuntimeException("Native inference library not available");
        Pointer handle = lib.thermastate_load_leaf(modelPath);
        if (handle == null) {
            throw new RuntimeException("Failed to load leaf model: " + modelPath);
        }
        return new LeafQNetwork(lib, handle);
    }

    /**
     * Decide best action for a subtree region.
     *
     * @param pdf       float[SMALL_PDF_SIZE] local data distribution (must sum to 1)
     * @param dataCount number of keys in this subtree
     * @return Decision with isLeaf=true (use leafType) or isLeaf=false (use fanout)
     */
    public Decision decide(float[] pdf, int dataCount) {
        if (pdf.length != SMALL_PDF_SIZE) {
            throw new IllegalArgumentException(
                "pdf length must be " + SMALL_PDF_SIZE + ", got " + pdf.length);
        }
        if (dataCount <= 0) {
            return new Decision(true, 0); // Cold leaf for empty
        }
        float[] outResult = new float[2];
        int rc = lib.thermastate_leaf_decide(handle, pdf, (float) dataCount, outResult);
        if (rc != 0) {
            throw new RuntimeException("Leaf inference failed, rc=" + rc);
        }
        boolean isLeaf = outResult[0] != 0.0f;
        int value = (int) (outResult[1] + 0.5f);
        return new Decision(isLeaf, value);
    }

    @Override
    public void close() {
        if (handle != null) {
            lib.thermastate_free_leaf(handle);
        }
    }
}
