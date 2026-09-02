/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Pointer;

/**
 * Wraps the native tree-plan generator (DARE + TSMDP).
 *
 * One JNA call per bulkLoad: send PDF+count per root bucket,
 * receive the full tree structure packed as int32[] (8 nodes per int).
 *
 * Packed format: 8 nodes per int32, 4 bits each (LSB first).
 *   action 0-2 = leaf (Cold/HotRead/HotWrite)
 *   action 3-10 = expand, fanout = 2^(action-2): 2,4,8,16,32,64,128,256
 *   0xF (15) = sentinel — end of data
 */
public final class TreePlan implements AutoCloseable {

    public static final int SMALL_PDF_SIZE = LeafQNetwork.SMALL_PDF_SIZE; // 1024

    private static final int SENTINEL = 0xF;
    private static final int NODES_PER_INT = 8;

    /** A single node in the tree plan. */
    public static final class Node {
        /** Raw action id: 0=Cold,1=HotRead,2=HotWrite; 3-10=expand. */
        public final int action;
        public final boolean isLeaf;

        Node(int action) {
            this.action = action;
            this.isLeaf = action <= 2;
        }

        /** Valid when isLeaf. 0=Cold, 1=HotRead, 2=HotWrite. */
        public int leafType() { return action; }

        /** Valid when !isLeaf. Expand fanout: 2,4,8,16,32,64,128,256. */
        public int fanout() { return 1 << (action - 2); }

        @Override
        public String toString() {
            return isLeaf
                ? ("Leaf(" + (action == 0 ? "Cold" : action == 1 ? "HotRead" : "HotWrite") + ")")
                : ("Inner(fanout=" + fanout() + ")");
        }
    }

    /** Per-bucket descriptor: where its subtree starts in the node array. */
    public static final class Bucket {
        public final int offset;  // start index in nodes[]
        public final int count;   // number of nodes for this bucket

        Bucket(int offset, int count) { this.offset = offset; this.count = count; }
    }

    private final ThermaStateInference lib;
    private final Pointer handle;
    private final int rootFanout;
    public final Node[] nodes;
    public final Bucket[] buckets;

    private TreePlan(ThermaStateInference lib, Pointer handle,
                     int rootFanout, Node[] nodes, Bucket[] buckets) {
        this.lib = lib;
        this.handle = handle;
        this.rootFanout = rootFanout;
        this.nodes = nodes;
        this.buckets = buckets;
    }

    public int rootFanout() { return rootFanout; }

    // Unpack int32[] → Node[]. 8 nodes per int, 4 bits each (LSB first). 0xF = sentinel.
    private static Node[] unpack(int[] packed, int totalNodes) {
        Node[] nodes = new Node[totalNodes];
        int ni = 0;
        outer:
        for (int p : packed) {
            for (int j = 0; j < NODES_PER_INT; j++) {
                int action = (p >>> (j * 4)) & SENTINEL;  // SENTINEL == 0xF
                if (action == SENTINEL || ni >= totalNodes) break outer;
                nodes[ni++] = new Node(action);
            }
        }
        return nodes;
    }

    /**
     * Generate a tree plan for a set of root buckets.
     *
     * @param modelPath   path to leaf_agent.pt, or "" for heuristic-only
     * @param fanoutTable float[256] DARE inner fanout table, or null for defaults
     * @param pdfs        float[numBuckets * SMALL_PDF_SIZE] one PDF per bucket
     * @param dataCounts  int[numBuckets] data count per bucket
     * @param lowers      double[numBuckets] lower bound per bucket
     * @param uppers      double[numBuckets] upper bound per bucket
     * @param globalLo    global key space lower bound
     * @param globalHi    global key space upper bound
     * @param fr          read frequency  [0,1] (0.5 for bulk load)
     * @param fw          write frequency [0,1] (0.5 for bulk load)
     * @param maxNodes    safety limit on output nodes
     */
    public static TreePlan generate(
            String modelPath,
            float[] fanoutTable,
            float rootFanout,
            float[] pdfs,
            int[] dataCounts,
            double[] lowers,
            double[] uppers,
            double globalLo,
            double globalHi,
            float fr, float fw,
            int maxNodes) {

        ThermaStateInference lib = ThermaStateInference.instance();
        if (lib == null) return null;
        Pointer handle = lib.thermastate_plan_create(
            modelPath, fanoutTable, rootFanout, globalLo, globalHi, fr, fw);
        if (handle == null) throw new RuntimeException("thermastate_plan_create failed");

        int numBuckets = dataCounts.length;
        int[] packed = new int[(maxNodes + NODES_PER_INT - 1) / NODES_PER_INT];
        int[] offsets = new int[numBuckets * 2];

        int nNodes = lib.thermastate_plan_generate(
            handle, pdfs, dataCounts, lowers, uppers,
            numBuckets, packed, maxNodes, offsets);
        if (nNodes < 0) throw new RuntimeException("thermastate_plan_generate failed");

        Node[] nodes = unpack(packed, nNodes);

        Bucket[] buckets = new Bucket[numBuckets];
        for (int b = 0; b < numBuckets; b++)
            buckets[b] = new Bucket(offsets[b * 2], offsets[b * 2 + 1]);

        return new TreePlan(lib, handle, (int)rootFanout, nodes, buckets);
    }

    @Override
    public void close() {
        if (handle != null) lib.thermastate_plan_free(handle);
    }
}
