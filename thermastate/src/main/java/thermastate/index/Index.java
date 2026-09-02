/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.sun.jna.Pointer;

import thermastate.agent.QNetwork;
import thermastate.agent.RouteAgent;
import thermastate.agent.ThermaStateInference;
import thermastate.agent.TreePlan;
import thermastate.monitor.TemperatureMonitor;
import thermastate.reorganizer.DriftDetector;
import thermastate.reorganizer.IntervalLock;

/**
 * Variable-depth learned index. Stores {@code double key → int bundleId} mappings.
 * Actual values live in {@code ValueBundleBlock<V>}; this tree stores only bundle IDs.
 *
 * Tree building: C++ recursively plans the complete tree using LeafAgentNetwork
 * (one JNA call), Java consumes the preorder PlanNode[] and inserts data.
 */
public class Index {

    private static final int PDF_SIZE = 1024;
    static final float DEFAULT_FR = 0.5f;
    static final float DEFAULT_FW = 0.5f;

    /** Hard ceiling on entries per leaf (incl. overflow chain). Above this the
     *  leaf is too large for its learned model / array layout, so the enclosing
     *  interval is forced through a LeafAgent-driven subtree reconstruction
     *  (split) instead of the unbounded {@link Leaf#expand()}. */
    public static final int MAX_LEAF_ENTRIES = 1024;

    private final double lower;
    private final double upper;
    private final Configuration conf;
    private InnerNode root;
    private final Map<Integer, LeafType> regionLeafType;
    private final java.util.Set<Node> forcedRebuilds = java.util.concurrent.ConcurrentHashMap.newKeySet();
    private TemperatureMonitor monitor;
    private String modelPath;
    private String globalModelPath;
    private String globalScalarPath;
    private DriftDetector driftDetector;
    private IntervalLock intervalLock;

    public Index(Configuration conf, double lower, double upper) {
        this(conf, lower, upper, null);
    }

    public Index(Configuration conf, double lower, double upper,
                 Map<Integer, LeafType> regionLeafType) {
        this.conf = conf;
        this.lower = lower;
        this.upper = upper;
        this.regionLeafType = regionLeafType;
        int rootFanout = (int) conf.rootFanOut;
        this.root = InnerNode.create(rootFanout, lower, upper);
    }

    public void setModelPath(String path) { this.modelPath = path; }
    public String getModelPath() { return modelPath; }

    public void setGlobalModelPath(String qPath, String scalarPath) {
        this.globalModelPath = qPath;
        this.globalScalarPath = scalarPath;
    }

    /** Compute global PDF (16384-bin) for Route Agent. */
    static float[] computeGlobalPDF(List<IntEntry> sortedData,
                                     double kMin, double kMax, int pdfSize) {
        float[] pdf = new float[pdfSize];
        int n = sortedData.size();
        if (n == 0 || kMax <= kMin) return pdf;
        double range = kMax - kMin;
        for (IntEntry e : sortedData) {
            int bin = (int) (pdfSize * (e.key - kMin) / range);
            if (bin < 0) bin = 0;
            if (bin >= pdfSize) bin = pdfSize - 1;
            pdf[bin] += 1.0f / n;
        }
        return pdf;
    }

    public void setMonitor(TemperatureMonitor monitor) { this.monitor = monitor; }
    public TemperatureMonitor getMonitor() { return monitor; }

    public void setDriftDetector(DriftDetector dd) { this.driftDetector = dd; }
    public DriftDetector getDriftDetector() { return driftDetector; }

    public void setIntervalLock(IntervalLock il) {
        this.intervalLock = il;
        if (il != null) registerIntervalRoots();
    }
    public IntervalLock getIntervalLock() { return intervalLock; }

    // ── Diagnostics: track which agents were used during bulkLoad ──

    private volatile boolean lastBulkLoadUsedRouteAgent;
    private volatile boolean lastBulkLoadUsedTreePlan;
    private volatile String lastBulkLoadSummary;

    public boolean lastBulkLoadUsedRouteAgent() { return lastBulkLoadUsedRouteAgent; }
    public boolean lastBulkLoadUsedTreePlan() { return lastBulkLoadUsedTreePlan; }
    public String lastBulkLoadSummary() { return lastBulkLoadSummary; }

    /** Register all routing nodes in the tree with the IntervalLock. */
    private void registerIntervalRoots() {
        if (intervalLock == null) return;
        registerRoutingNodes(root);
    }

    private void registerRoutingNodes(Node node) {
        if (node == null || !isRoutingNode(node)) return;
        intervalLock.register(node);
        int nc = routingChildCount(node);
        for (int i = 0; i < nc; i++) {
            Node child = routingGetChild(node, i);
            if (child != null) registerRoutingNodes(child);
        }
    }
    public Configuration getConfig() { return conf; }
    public double lower() { return lower; }
    public double upper() { return upper; }
    public InnerNode root() { return root; }

    static LeafType leafTypeFromAction(int action) {
        switch (action) {
            case 0:  return LeafType.COLD;
            case 1:  return LeafType.HOT_READ;
            case 2:  return LeafType.HOT_WRITE;
            default: return LeafType.COLD;
        }
    }

    public static Leaf createLeaf(LeafType type, int capacity, double lower, double upper) {
        switch (type) {
            case HOT_READ:  return HotReadLeaf.create(capacity, lower, upper);
            case HOT_WRITE: return HotWriteLeaf.create(capacity, lower, upper);
            case COLD:
            default:        return ColdLeaf.create(capacity, lower, upper);
        }
    }

    public static void bulkInsertInto(Leaf leaf, double key, int bundleId) {
        if (leaf instanceof HotReadLeaf) {
            ((HotReadLeaf) leaf).bulkInsert(key, bundleId);
        } else if (leaf instanceof HotWriteLeaf) {
            ((HotWriteLeaf) leaf).bulkInsert(key, bundleId);
        } else if (leaf instanceof ColdLeaf) {
            ((ColdLeaf) leaf).bulkInsert(key, bundleId);
        } else {
            leaf.put(key, bundleId);
        }
    }

    public static float[] computeLocalPDF(List<IntEntry> entries, int from, int to,
                                           double lo, double hi) {
        float[] pdf = new float[PDF_SIZE];
        int n = to - from;
        if (n == 0 || hi <= lo) return pdf;

        double range = hi - lo;
        for (int i = from; i < to; i++) {
            int bin = (int) ((entries.get(i).key - lo) / range * PDF_SIZE);
            if (bin < 0) bin = 0;
            if (bin >= PDF_SIZE) bin = PDF_SIZE - 1;
            pdf[bin] += 1.0f / n;
        }
        return pdf;
    }

    private static float[] flattenFanouts(Configuration conf) {
        float[] table = new float[Configuration.INNER_FANOUT_SIZE];
        for (int j = 0; j < Configuration.INNER_FANOUT_COLUMN; j++) {
            table[j] = conf.fanOuts[0][j];
        }
        return table;
    }

    // ── Routing helpers ──

    private static boolean isRoutingNode(Node node) {
        return node instanceof InnerNode || node instanceof ColdInnerNode;
    }

    private static Node routingGetChild(Node node, int pos) {
        if (node instanceof InnerNode) return ((InnerNode) node).getChild(pos);
        return ((ColdInnerNode) node).getChild(pos);
    }

    private static void routingSetChild(Node node, int pos, Node child) {
        if (node instanceof InnerNode) ((InnerNode) node).setChild(pos, child);
        else ((ColdInnerNode) node).setChild(pos, child);
    }

    private static int routingChildCount(Node node) {
        if (node instanceof InnerNode) return ((InnerNode) node).childCount();
        return ((ColdInnerNode) node).childCount();
    }

    // ── Tree building ──

    private Node buildPlanTree(TreePlan.Node[] planNodes, int[] posRef,
                                double lo, double hi) {
        TreePlan.Node pn = planNodes[posRef[0]++];
        if (pn.isLeaf) {
            int cap = Math.max(DataNode.DATA_NODE_SIZE * 4, 32);
            return createLeaf(leafTypeFromAction(pn.action), cap, lo, hi);
        } else {
            int fanout = pn.fanout();
            InnerNode inner = InnerNode.create(fanout, lo, hi);
            for (int i = 0; i < fanout; i++) {
                double[] ci = inner.subInterval(i);
                Node child = buildPlanTree(planNodes, posRef, ci[0], ci[1]);
                inner.setChild(i, child);
            }
            return inner;
        }
    }

    private static void insertIntoTree(Node root, double key, int bundleId) {
        Node node = root;
        while (isRoutingNode(node)) {
            int pos = node.forward(key);
            node = routingGetChild(node, pos);
        }
        Leaf leaf = (Leaf) node;
        bulkInsertInto(leaf, key, bundleId);
    }

    /**
     * Rebuild the subtree rooted at {@code intervalRoot} into a fresh subtree,
     * re-partitioning its entries across a data-driven fanout. This is the
     * subtree-reconstruction step of the paper's three-stage pipeline: snapshot
     * the interval, sort, re-partition into a new InnerNode whose leaf children
     * are recreated from the snapshot, and return the new root.
     *
     * The caller holds the Interval Lock for the interval and atomically swaps
     * the returned root into place via {@link #replaceChild}.
     */
    public Node rebuildSubtree(Node intervalRoot) {
        List<IntEntry> entries = new ArrayList<>();
        collectSubtreeEntries(intervalRoot, entries);
        if (entries.isEmpty()) return intervalRoot;

        entries.sort(Comparator.comparingDouble(e -> e.key));

        int dc = entries.size();
        int fanout;
        if (dc <= 10)          fanout = 1;
        else if (dc <= 50)     fanout = 2;
        else if (dc <= 200)    fanout = 4;
        else if (dc <= 1000)   fanout = 8;
        else if (dc <= 5000)   fanout = 16;
        else if (dc <= 20000)  fanout = 32;
        else if (dc <= 100000) fanout = 64;
        else                   fanout = 128;

        // Never shrink below the current fanout — avoids collapsing an
        // already-deep interval and preserves split-only growth semantics.
        fanout = Math.max(fanout, intervalRoot.capacity());

        InnerNode newRoot = InnerNode.create(fanout, intervalRoot.lower(), intervalRoot.upper());

        int[] buckets = new int[fanout];
        for (IntEntry e : entries) {
            int pos = newRoot.forward(e.key);
            buckets[Math.min(fanout - 1, Math.max(0, pos))]++;
        }

        // LeafAgent-driven recursive plan when the model is available
        if (modelPath != null && !modelPath.isEmpty()) {
            Node planned = buildPlannedSubtree(entries, buckets, fanout, newRoot);
            if (planned != null) return planned;
        }

        int left = 0;
        for (int i = 0; i < fanout; i++) {
            int right = left + buckets[i];
            if (buckets[i] == 0) {
                left = right;
                continue;
            }
            double[] sub = newRoot.subInterval(i);
            int cap = Math.max(DataNode.DATA_NODE_SIZE * 4, buckets[i] * 2);
            Leaf leaf = createLeaf(LeafType.COLD, cap, sub[0], sub[1]);
            for (int k = left; k < right; k++) {
                IntEntry e = entries.get(k);
                bulkInsertInto(leaf, e.key, e.bundleId);
            }
            newRoot.setLeafChild(i, leaf);
            left = right;
        }

        return newRoot;
    }

    /**
     * Build the rebuilt subtree from a LeafAgent (TreePlan) recursive plan.
     * Returns null to signal the caller to fall back to the heuristic layout.
     */
    private Node buildPlannedSubtree(List<IntEntry> entries, int[] buckets,
                                     int fanout, InnerNode newRoot) {
        try {
            int numBuckets = 0;
            for (int b : buckets) if (b > 0) numBuckets++;

            float[] allPdfs = new float[numBuckets * PDF_SIZE];
            int[] dataCounts = new int[numBuckets];
            double[] lowers = new double[numBuckets];
            double[] uppers = new double[numBuckets];

            int left = 0, bi = 0;
            for (int i = 0; i < fanout; i++) {
                int right = left + buckets[i];
                if (buckets[i] == 0) { left = right; continue; }
                double[] sub = newRoot.subInterval(i);
                float[] pdf = computeLocalPDF(entries, left, right, sub[0], sub[1]);
                System.arraycopy(pdf, 0, allPdfs, bi * PDF_SIZE, PDF_SIZE);
                dataCounts[bi] = buckets[i];
                lowers[bi] = sub[0];
                uppers[bi] = sub[1];
                bi++;
                left = right;
            }

            TreePlan plan = TreePlan.generate(
                modelPath, flattenFanouts(conf), fanout,
                allPdfs, dataCounts, lowers, uppers,
                newRoot.lower(), newRoot.upper(),
                DEFAULT_FR, DEFAULT_FW, 500_000);
            if (plan == null) return null;

            try {
                left = 0; bi = 0;
                for (int i = 0; i < fanout; i++) {
                    int right = left + buckets[i];
                    if (buckets[i] == 0) { left = right; continue; }
                    TreePlan.Bucket bucket = plan.buckets[bi];
                    int[] posRef = new int[]{bucket.offset};
                    double[] sub = newRoot.subInterval(i);
                    Node subtree = buildPlanTree(plan.nodes, posRef, sub[0], sub[1]);
                    newRoot.setChild(i, subtree);
                    for (int k = left; k < right; k++) {
                        IntEntry e = entries.get(k);
                        insertIntoTree(subtree, e.key, e.bundleId);
                    }
                    bi++;
                    left = right;
                }
                return newRoot;
            } finally {
                plan.close();
            }
        } catch (Exception e) {
            // Native inference unavailable — fall back to heuristic layout.
            return null;
        }
    }

    private void collectSubtreeEntries(Node node, List<IntEntry> out) {
        if (node == null) return;
        if (node instanceof InnerNode) {
            InnerNode inner = (InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) {
                Node child = inner.getChild(i);
                if (child != null) collectSubtreeEntries(child, out);
            }
        } else if (node instanceof ColdInnerNode) {
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child != null) collectSubtreeEntries(child, out);
            }
        } else if (node instanceof Leaf) {
            Leaf leaf = (Leaf) node;
            leaf.forEachEntry((k, v) -> out.add(new IntEntry(k, v)));
        }
    }

    // ── bulkLoad ─────────────────────────────────────────────────────

    /**
     * Build the index from sorted (key, bundleId) data.
     */
    public void bulkLoad(List<IntEntry> sortedData) {
        // Phase 1: Route Agent GA search (with graceful fallback)
        int rootFanout = root.capacity();
        float[] dareFanOuts = flattenFanouts(conf);
        boolean routeAgentOk = false;

        if (globalModelPath != null && !globalModelPath.isEmpty()
                && globalScalarPath != null && !globalScalarPath.isEmpty()) {
            ThermaStateInference lib = ThermaStateInference.instance();
            if (lib != null) {
                Pointer handle = lib.thermastate_load(globalModelPath, globalScalarPath);
                if (handle != null) {
                    try {
                        float[] globalPdf = computeGlobalPDF(
                            sortedData, lower, upper, QNetwork.PDF_SIZE);
                        RouteAgent.RouteConfig rc = RouteAgent.search(
                            lib, handle, globalPdf, (float) sortedData.size());
                        rootFanout = rc.rootFanout;
                        dareFanOuts = rc.fanOuts;
                        routeAgentOk = true;
                    } catch (Exception e) {
                        System.err.println("[Index] Route Agent inference failed: "
                            + e.getMessage());
                    } finally {
                        lib.thermastate_free(handle);
                    }
                }
            }
        }
        lastBulkLoadUsedRouteAgent = routeAgentOk;
        if (routeAgentOk) {
            this.root = InnerNode.create(rootFanout, lower, upper);
        }

        // Phase 2: Partition
        int[] rootBuckets = new int[rootFanout];
        for (IntEntry e : sortedData) {
            rootBuckets[root.forward(e.key)]++;
        }

        int numBuckets = 0;
        for (int i = 0; i < rootFanout; i++) {
            if (rootBuckets[i] > 0) numBuckets++;
        }
        if (numBuckets == 0) return;

        float[] allPdfs    = new float[numBuckets * PDF_SIZE];
        int[]   dataCounts = new int[numBuckets];
        double[] lowers    = new double[numBuckets];
        double[] uppers    = new double[numBuckets];

        int left = 0;
        int bi = 0;
        for (int rootIdx = 0; rootIdx < rootFanout; rootIdx++) {
            int right = left + rootBuckets[rootIdx];
            if (rootBuckets[rootIdx] > 0) {
                double[] interval = root.subInterval(rootIdx);
                float[] pdf = computeLocalPDF(sortedData, left, right, interval[0], interval[1]);
                System.arraycopy(pdf, 0, allPdfs, bi * PDF_SIZE, PDF_SIZE);
                dataCounts[bi] = rootBuckets[rootIdx];
                lowers[bi] = interval[0];
                uppers[bi] = interval[1];
                bi++;
            }
            left = right;
        }

        // Phase 3: Tree plan via LeafAgent
        TreePlan plan = null;
        if (modelPath != null && !modelPath.isEmpty()) {
            plan = TreePlan.generate(
                modelPath, dareFanOuts, (float) rootFanout,
                allPdfs, dataCounts, lowers, uppers,
                lower, upper, DEFAULT_FR, DEFAULT_FW, 500_000);
            lastBulkLoadUsedTreePlan = (plan != null);
        } else {
            lastBulkLoadUsedTreePlan = false;
        }

        // Phase 4: Build tree and insert data
        left = 0;
        bi = 0;
        for (int rootIdx = 0; rootIdx < rootFanout; rootIdx++) {
            int right = left + rootBuckets[rootIdx];
            if (rootBuckets[rootIdx] == 0) {
                left = right;
                continue;
            }

            Node subtree;
            if (plan != null) {
                TreePlan.Bucket bucket = plan.buckets[bi];
                int[] posRef = new int[]{bucket.offset};
                double[] interval = root.subInterval(rootIdx);
                subtree = buildPlanTree(plan.nodes, posRef, interval[0], interval[1]);
            } else {
                // Heuristic fallback
                double[] interval = root.subInterval(rootIdx);
                int dc = rootBuckets[rootIdx];
                int fanout;
                if (dc <= 10)           fanout = 1;
                else if (dc <= 50)      fanout = 2;
                else if (dc <= 200)     fanout = 4;
                else if (dc <= 1000)    fanout = 8;
                else if (dc <= 5000)    fanout = 16;
                else if (dc <= 20000)   fanout = 32;
                else if (dc <= 100000)  fanout = 64;
                else                    fanout = 128;

                InnerNode inner = InnerNode.create(fanout, interval[0], interval[1]);
                int[] innerBuckets = new int[fanout];
                for (int k = left; k < right; k++) {
                    innerBuckets[inner.forward(sortedData.get(k).key)]++;
                }
                int il = left;
                for (int innerIdx = 0; innerIdx < fanout; innerIdx++) {
                    int ir = il + innerBuckets[innerIdx];
                    if (innerBuckets[innerIdx] > 0) {
                        double[] di = inner.subInterval(innerIdx);
                        int cap = Math.max(DataNode.DATA_NODE_SIZE * 4, innerBuckets[innerIdx] * 2);
                        Leaf leaf = createLeaf(LeafType.COLD, cap, di[0], di[1]);
                        for (int k = il; k < ir; k++) {
                            IntEntry e = sortedData.get(k);
                            bulkInsertInto(leaf, e.key, e.bundleId);
                        }
                        inner.setLeafChild(innerIdx, leaf);
                    }
                    il = ir;
                }
                subtree = inner;
            }

            root.setChild(rootIdx, subtree);
            if (plan != null) {
                for (int k = left; k < right; k++) {
                    IntEntry e = sortedData.get(k);
                    insertIntoTree(subtree, e.key, e.bundleId);
                }
            }

            left = right;
            bi++;
        }

        if (plan != null) plan.close();

        // Phase 5: Temperature labeling
        labelAndConvert(root);

        // Phase 6: Register drift regions
        registerDriftRegions();

        // Diagnostic summary
        lastBulkLoadSummary = String.format(
            "bulkLoad: %d entries, rootFanout=%d, routeAgent=%s, treePlan=%s, stats=%s",
            sortedData.size(), rootFanout,
            lastBulkLoadUsedRouteAgent ? "YES" : "no",
            lastBulkLoadUsedTreePlan ? "YES" : "no",
            getStats());
    }

    // ── Temperature labeling ──

    private void labelAndConvert(InnerNode rootNode) {
        final double kRange = upper - lower;
        if (kRange <= 0) return;

        final double BETA = 0.5;
        final double TAU  = 0.3;

        List<PositionedNode> routingNodes = new ArrayList<>();
        collectRoutingNodes(rootNode, routingNodes);

        if (routingNodes.size() <= 1) return;

        double maxFreq = 0;
        for (PositionedNode pn : routingNodes) {
            pn.freq = routingChildCount(pn.node);
            if (pn.freq > maxFreq) maxFreq = pn.freq;
        }
        if (maxFreq == 0) return;

        for (PositionedNode pn : routingNodes) {
            if (pn.parent == null) continue;

            double A = Math.log(1.0 + pn.freq) / Math.log(1.0 + maxFreq);
            double R = (pn.node.upper() - pn.node.lower()) / kRange;
            double T = A * (1.0 + BETA * R);
            boolean isHot = T >= TAU;

            if (!isHot && pn.node instanceof InnerNode) {
                InnerNode inner = (InnerNode) pn.node;
                ColdInnerNode cold = ColdInnerNode.create(
                    inner.capacity(), inner.lower(), inner.upper());
                for (int i = 0; i < inner.capacity(); i++) {
                    Node child = inner.getChild(i);
                    if (child != null) cold.setChild(i, child);
                }
                routingSetChild(pn.parent, pn.position, cold);
            }
        }
    }

    private static class PositionedNode {
        final Node node;
        final Node parent;
        final int position;
        double freq;

        PositionedNode(Node node, Node parent, int position) {
            this.node = node;
            this.parent = parent;
            this.position = position;
        }
    }

    private static void collectRoutingNodes(Node node, List<PositionedNode> out) {
        collectRoutingNodes(node, null, -1, out);
    }

    private static void collectRoutingNodes(Node node, Node parent, int pos,
                                             List<PositionedNode> out) {
        if (node == null) return;
        if (node instanceof InnerNode) {
            out.add(new PositionedNode(node, parent, pos));
            InnerNode inner = (InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) {
                Node child = inner.getChild(i);
                if (child != null) collectRoutingNodes(child, node, i, out);
            }
        } else if (node instanceof ColdInnerNode) {
            out.add(new PositionedNode(node, parent, pos));
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child != null) collectRoutingNodes(child, node, i, out);
            }
        }
    }

    // ── Tree navigation ──

    public static final class ParentRef {
        public final Node parent;
        public final int position;

        public ParentRef(Node parent, int position) {
            this.parent = parent;
            this.position = position;
        }
    }

    /** Result from routing: leaf + its direct parent in the tree. */
    static final class RouteResult {
        final Leaf leaf;
        final Node parent;
        final int position;

        RouteResult(Leaf leaf, Node parent, int position) {
            this.leaf = leaf;
            this.parent = parent;
            this.position = position;
        }
    }

    public ParentRef findParent(Node target) {
        if (target == root) return null;
        return findParentRec(root, target);
    }

    private ParentRef findParentRec(Node node, Node target) {
        if (!isRoutingNode(node)) return null;
        int nc = routingChildCount(node);
        for (int i = 0; i < nc; i++) {
            Node child = routingGetChild(node, i);
            if (child == null) continue;
            if (child == target) return new ParentRef(node, i);
            if (isRoutingNode(child)) {
                ParentRef found = findParentRec(child, target);
                if (found != null) return found;
            }
        }
        return null;
    }

    public void replaceChild(Node parent, int position, Node newChild) {
        if (parent instanceof InnerNode) {
            ((InnerNode) parent).setChild(position, newChild);
        } else if (parent instanceof ColdInnerNode) {
            ((ColdInnerNode) parent).setChild(position, newChild);
        }
    }

    /**
     * Re-route to the leaf at a known parent + position.
     * Returns null if the slot is empty or points to a routing node.
     * Used by Reorganizer to verify leaf identity after re-acquiring writeLock.
     */
    public Leaf routeToLeafForRef(ParentRef ref) {
        Node child = routingGetChild(ref.parent, ref.position);
        if (child != null && !isRoutingNode(child)) return (Leaf) child;
        return null;
    }

    /** Re-route to the child at a known parent + position (leaf or routing node). */
    public Node childAt(ParentRef ref) {
        return routingGetChild(ref.parent, ref.position);
    }

    /** Mark an interval root for a size-driven forced subtree reconstruction. */
    public void forceRebuild(Node intervalRoot) {
        forcedRebuilds.add(intervalRoot);
    }

    /** Drain and clear the set of intervals flagged for forced reconstruction. */
    public java.util.Set<Node> drainForcedRebuilds() {
        if (forcedRebuilds.isEmpty()) return java.util.Set.of();
        java.util.Set<Node> out = new java.util.HashSet<>(forcedRebuilds);
        forcedRebuilds.clear();
        return out;
    }

    public void replaceRoot(InnerNode newRoot) {
        this.root = newRoot;
    }

    // ── Drift Detection ──

    private void registerDriftRegions() {
        if (driftDetector == null) return;

        List<PositionedNode> routingNodes = new ArrayList<>();
        collectRoutingNodes(root, routingNodes);

        for (PositionedNode pn : routingNodes) {
            if (pn.node == root) continue;
            Set<Leaf> leaves = new HashSet<>();
            collectSubtreeLeaves(pn.node, leaves);
            if (!leaves.isEmpty()) {
                driftDetector.registerRegion(pn.node, leaves);
            }
        }
    }

    private static void collectSubtreeLeaves(Node node, Set<Leaf> out) {
        if (node == null) return;
        if (!isRoutingNode(node)) {
            out.add((Leaf) node);
            return;
        }
        int nc = routingChildCount(node);
        for (int i = 0; i < nc; i++) {
            Node child = routingGetChild(node, i);
            if (child != null) collectSubtreeLeaves(child, out);
        }
    }

    // ── CRUD ──

    private Leaf routeToLeaf(double key) {
        Node node = root;
        int depth = 0;
        while (isRoutingNode(node)) {
            if (monitor != null && node instanceof InnerNode) {
                monitor.recordRoute((InnerNode) node, depth);
            }
            int pos = node.forward(key);
            node = routingGetChild(node, pos);
            if (node == null) return null;
            depth++;
        }
        return (Leaf) node;
    }

    /**
     * Route to a leaf, tracking the parent routing node and position.
     * Returns null if the tree has insufficient depth to find a leaf.
     */
    private RouteResult routeToLeafWithParent(double key) {
        Node node = root;
        int depth = 0;
        Node parent = root;
        int position = -1;
        while (isRoutingNode(node)) {
            if (monitor != null && node instanceof InnerNode) {
                monitor.recordRoute((InnerNode) node, depth);
            }
            position = node.forward(key);
            Node child = routingGetChild(node, position);
            if (child == null) {
                return new RouteResult(null, node, position);
            }
            if (!isRoutingNode(child)) {
                return new RouteResult((Leaf) child, node, position);
            }
            parent = child;
            node = child;
            depth++;
        }
        return new RouteResult((Leaf) node, parent, position);
    }

    /** Re-route to leaf directly under a known parent, returning the current leaf ref. */
    private Leaf routeToLeafUnderParent(double key, Node parent, int position) {
        int pos = parent.forward(key);
        Node child = routingGetChild(parent, pos);
        if (child != null && !isRoutingNode(child)) {
            return (Leaf) child;
        }
        return null;
    }

    // ── Lock-aware CRUD ──

    /** Returns bundleId or -1 if not found. */
    public int get(double key) {
        if (intervalLock != null) {
            RouteResult rr = routeToLeafWithParent(key);
            if (rr == null || rr.leaf == null) return -1;
            intervalLock.readLock(rr.parent);
            try {
                int result = rr.leaf.get(key);
                if (monitor != null && result >= 0) monitor.recordRead(rr.leaf);
                return result;
            } finally {
                intervalLock.readUnlock(rr.parent);
            }
        }
        // Lock-free fallback
        Leaf leaf = routeToLeaf(key);
        if (leaf == null) return -1;
        int result = leaf.get(key);
        if (monitor != null && result >= 0) monitor.recordRead(leaf);
        return result;
    }

    public boolean put(double key, int bundleId) {
        if (intervalLock != null) {
            return putLocked(key, bundleId);
        }
        return putInternal(key, bundleId);
    }

    /** Lock-protected put. Escalates from readLock to writeLock for expand/gap cases. */
    private boolean putLocked(double key, int bundleId) {
        while (true) {
            RouteResult rr = routeToLeafWithParent(key);
            if (rr == null) return false;

            if (rr.leaf != null) {
                // Normal case: leaf exists
                intervalLock.readLock(rr.parent);
                boolean readLockHeld = true;
                try {
                    // Single-node ceiling: independent of isFull, since expand
                    // grows capacity faster than size and may keep a leaf under
                    // the fill threshold long past the ceiling.
                    if (rr.leaf.size() > MAX_LEAF_ENTRIES
                            || (rr.leaf.isFull() && rr.leaf.totalEntries() > MAX_LEAF_ENTRIES)) {
                        // Over the ceiling — flag the enclosing interval for a
                        // LeafAgent-driven split; keep serving writes via the
                        // overflow chain until the reorganizer rebuilds it.
                        forceRebuild(rr.parent);
                    } else if (rr.leaf.pendingDelta == null && rr.leaf.isFull()) {
                        intervalLock.readUnlock(rr.parent);
                        readLockHeld = false;
                        // Escalate to writeLock for expand
                        intervalLock.writeLock(rr.parent);
                        try {
                            Leaf current = routeToLeafUnderParent(key, rr.parent, rr.position);
                            if (current == rr.leaf) {
                                Leaf expanded = rr.leaf.expand();
                                replaceChild(rr.parent, rr.position, expanded);
                            }
                        } finally {
                            intervalLock.writeUnlock(rr.parent);
                        }
                        continue;
                    }
                    boolean ok = rr.leaf.put(key, bundleId);
                    if (monitor != null && ok) monitor.recordWrite(rr.leaf);
                    return ok;
                } finally {
                    if (readLockHeld) intervalLock.readUnlock(rr.parent);
                }
            } else {
                // Tree gap — structural change under writeLock
                intervalLock.writeLock(rr.parent);
                try {
                    return putInternalAt(key, bundleId, rr.parent, rr.position);
                } finally {
                    intervalLock.writeUnlock(rr.parent);
                }
            }
        }
    }

    /** Original put logic, extracted for both locked and lock-free paths. */
    private boolean putInternal(double key, int bundleId) {
        Leaf leaf = routeToLeaf(key);
        if (leaf == null) {
            int rootPos = root.forward(key);
            Node child = root.getChild(rootPos);
            if (child == null) {
                double[] interval = root.subInterval(rootPos);
                int fanout = conf.getInnerFanout(rootPos);
                InnerNode inner = InnerNode.create(fanout, interval[0], interval[1]);
                root.setInnerChild(rootPos, inner);
                int innerPos = inner.forward(key);
                double[] leafInterval = inner.subInterval(innerPos);
                int cap = DataNode.DATA_NODE_SIZE * 4;
                Leaf newLeaf = createLeaf(LeafType.COLD, cap, leafInterval[0], leafInterval[1]);
                inner.setLeafChild(innerPos, newLeaf);
                leaf = newLeaf;
            } else if (isRoutingNode(child)) {
                int innerPos = child.forward(key);
                double[] leafInterval;
                if (child instanceof InnerNode) {
                    leafInterval = ((InnerNode) child).subInterval(innerPos);
                } else {
                    leafInterval = ((ColdInnerNode) child).subInterval(innerPos);
                }
                int cap = DataNode.DATA_NODE_SIZE * 4;
                Leaf newLeaf = createLeaf(LeafType.COLD, cap, leafInterval[0], leafInterval[1]);
                routingSetChild(child, innerPos, newLeaf);
                leaf = newLeaf;
            } else {
                leaf = (Leaf) child;
            }
        }

        if (leaf.isFull()) {
            leaf = leaf.expand();
            Node node = root;
            while (isRoutingNode(node)) {
                int pos = node.forward(key);
                Node next = routingGetChild(node, pos);
                if (next == leaf) {
                    routingSetChild(node, pos, leaf);
                    break;
                }
                node = next;
            }
        }

        boolean ok = leaf.put(key, bundleId);
        if (monitor != null && ok) monitor.recordWrite(leaf);
        return ok;
    }

    /** Structural put into a known parent slot (gap fill). Caller holds writeLock. */
    private boolean putInternalAt(double key, int bundleId,
                                   Node parent, int position) {
        Node child = routingGetChild(parent, position);
        Leaf leaf;
        if (child == null) {
            double[] interval;
            if (parent instanceof InnerNode) {
                interval = ((InnerNode) parent).subInterval(position);
            } else {
                interval = ((ColdInnerNode) parent).subInterval(position);
            }
            int cap = DataNode.DATA_NODE_SIZE * 4;
            leaf = createLeaf(LeafType.COLD, cap, interval[0], interval[1]);
            routingSetChild(parent, position, leaf);
        } else if (isRoutingNode(child)) {
            int innerPos = child.forward(key);
            double[] leafInterval;
            if (child instanceof InnerNode) {
                leafInterval = ((InnerNode) child).subInterval(innerPos);
            } else {
                leafInterval = ((ColdInnerNode) child).subInterval(innerPos);
            }
            int cap = DataNode.DATA_NODE_SIZE * 4;
            leaf = createLeaf(LeafType.COLD, cap, leafInterval[0], leafInterval[1]);
            routingSetChild(child, innerPos, leaf);
        } else {
            leaf = (Leaf) child;
        }

        if (leaf.isFull()) {
            leaf = leaf.expand();
            replaceChild(parent, position, leaf);
        }

        boolean ok = leaf.put(key, bundleId);
        if (monitor != null && ok) monitor.recordWrite(leaf);
        return ok;
    }

    public boolean erase(double key) {
        if (intervalLock != null) {
            RouteResult rr = routeToLeafWithParent(key);
            if (rr == null || rr.leaf == null) return false;

            intervalLock.readLock(rr.parent);
            boolean readLockHeld = true;
            try {
                if (rr.leaf instanceof DataNode && ((DataNode) rr.leaf).isUnderMin()) {
                    intervalLock.readUnlock(rr.parent);
                    readLockHeld = false;
                    intervalLock.writeLock(rr.parent);
                    try {
                        Leaf current = routeToLeafUnderParent(key, rr.parent, rr.position);
                        if (current == rr.leaf) {
                            DataNode dn = DataNode.shrink((DataNode) rr.leaf);
                            replaceChild(rr.parent, rr.position, dn);
                            return dn.erase(key);
                        }
                        if (current != null) return current.erase(key);
                        return false;
                    } finally {
                        intervalLock.writeUnlock(rr.parent);
                    }
                }
                return rr.leaf.erase(key);
            } finally {
                if (readLockHeld) intervalLock.readUnlock(rr.parent);
            }
        }
        // Lock-free fallback
        Leaf leaf = routeToLeaf(key);
        if (leaf == null) return false;

        if (leaf instanceof DataNode && ((DataNode) leaf).isUnderMin()) {
            DataNode dn = DataNode.shrink((DataNode) leaf);
            Node node = root;
            while (isRoutingNode(node)) {
                int pos = node.forward(key);
                if (routingGetChild(node, pos) == leaf) {
                    routingSetChild(node, pos, dn);
                    break;
                }
                node = routingGetChild(node, pos);
            }
            leaf = dn;
        }

        return leaf.erase(key);
    }

    /** Returns old bundleId or -1. */
    public int update(double key, int bundleId) {
        if (intervalLock != null) {
            RouteResult rr = routeToLeafWithParent(key);
            if (rr == null || rr.leaf == null) return -1;

            intervalLock.readLock(rr.parent);
            try {
                int old = rr.leaf.get(key);
                if (old < 0) return -1;

                if (monitor != null) {
                    monitor.recordRead(rr.leaf);
                    monitor.recordWrite(rr.leaf);
                }

                rr.leaf.erase(key);
                rr.leaf.put(key, bundleId);
                return old;
            } finally {
                intervalLock.readUnlock(rr.parent);
            }
        }
        // Lock-free fallback
        Leaf leaf = routeToLeaf(key);
        if (leaf == null) return -1;

        int old = leaf.get(key);
        if (old < 0) return -1;

        if (monitor != null) {
            monitor.recordRead(leaf);
            monitor.recordWrite(leaf);
        }

        leaf.erase(key);
        leaf.put(key, bundleId);
        return old;
    }

    // ── Traversal ──

    public void forEach(IntEntryVisitor visitor) {
        forEachNode(root, visitor);
    }

    private void forEachNode(Node node, IntEntryVisitor visitor) {
        if (node instanceof InnerNode) {
            InnerNode inner = (InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) {
                if (inner.isInnerSlot(i)) {
                    Node child = inner.getInnerChild(i);
                    if (child != null) forEachNode(child, visitor);
                } else {
                    Leaf leaf = inner.getLeafChild(i);
                    if (leaf != null) leaf.forEachEntry(visitor::visit);
                }
            }
        } else if (node instanceof ColdInnerNode) {
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child instanceof Leaf) {
                    Leaf leaf = (Leaf) child;
                    leaf.forEachEntry(visitor::visit);
                } else if (child != null) {
                    forEachNode(child, visitor);
                }
            }
        }
    }

    public int size() {
        int[] count = {0};
        forEach((k, v) -> count[0]++);
        return count[0];
    }

    // ── Stats ──

    public static class IndexStats {
        public int innerNodeCount;
        public int dataNodeCount;
        public int maxDepth;
        public int totalEntries;
        public double minDensity = Double.MAX_VALUE;
        public double maxDensity;
        public double avgDensity;

        @Override
        public String toString() {
            return String.format(
                "innerNodes=%d dataNodes=%d maxDepth=%d entries=%d density=[%.3f, %.3f] avg=%.3f",
                innerNodeCount, dataNodeCount, maxDepth, totalEntries,
                minDensity, maxDensity, avgDensity);
        }
    }

    public IndexStats getStats() {
        IndexStats stats = new IndexStats();
        stats.innerNodeCount = 1;
        collectStats(root, 1, stats);
        if (stats.dataNodeCount > 0) {
            stats.avgDensity /= stats.dataNodeCount;
        } else {
            stats.minDensity = 0;
        }
        return stats;
    }

    private void collectStats(Node node, int depth, IndexStats stats) {
        stats.maxDepth = Math.max(stats.maxDepth, depth);
        if (node instanceof InnerNode) {
            InnerNode inner = (InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) {
                if (inner.isInnerSlot(i)) {
                    Node child = inner.getInnerChild(i);
                    if (child != null) {
                        stats.innerNodeCount++;
                        collectStats(child, depth + 1, stats);
                    }
                } else {
                    Leaf leaf = inner.getLeafChild(i);
                    if (leaf == null) continue;
                    stats.dataNodeCount++;
                    double density = (double) leaf.size() / leaf.capacity();
                    stats.minDensity = Math.min(stats.minDensity, density);
                    stats.maxDensity = Math.max(stats.maxDensity, density);
                    stats.avgDensity += density;
                    stats.totalEntries += leaf.size();
                }
            }
        } else if (node instanceof ColdInnerNode) {
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child instanceof Leaf) {
                    Leaf leaf = (Leaf) child;
                    stats.dataNodeCount++;
                    double density = (double) leaf.size() / leaf.capacity();
                    stats.minDensity = Math.min(stats.minDensity, density);
                    stats.maxDensity = Math.max(stats.maxDensity, density);
                    stats.avgDensity += density;
                    stats.totalEntries += leaf.size();
                } else if (child != null) {
                    stats.innerNodeCount++;
                    collectStats(child, depth + 1, stats);
                }
            }
        }
    }

    // ── Tree dump ──

    public void dumpTree() { dumpTree(System.out); }

    public void dumpTree(PrintStream out) {
        out.printf("=== Index Tree (rootFanout=%d, range=[%.0f, %.0f]) ===%n",
                   root.capacity(), lower, upper);
        dumpInner(out, root, 0, "");
        out.println();
    }

    private void dumpInner(PrintStream out, Node node, int depth, String prefix) {
        if (node instanceof InnerNode) {
            dumpInnerNode(out, (InnerNode) node, depth, prefix, "HotInnerNode");
        } else if (node instanceof ColdInnerNode) {
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child == null) continue;
                double[] iv = cold.subInterval(i);
                if (cold.isInnerSlot(i)) {
                    out.printf("%s├─[L%d slot %d] ColdInnerNode(fanout=%d) range=[%.0f,%.0f)%n",
                               prefix, depth, i, child.capacity(), iv[0], iv[1]);
                    dumpInner(out, child, depth + 1, prefix + "│  ");
                } else {
                    Leaf leaf = cold.getLeafChild(i);
                    out.printf("%s├─[L%d slot %d] %s size=%d range=[%.0f,%.0f)%n",
                               prefix, depth, i, leaf.getClass().getSimpleName(),
                               leaf.size(), iv[0], iv[1]);
                }
            }
        }
    }

    private void dumpInnerNode(PrintStream out, InnerNode inner, int depth,
                                String prefix, String label) {
        for (int i = 0; i < inner.capacity(); i++) {
            Node child = inner.getChild(i);
            if (child == null) continue;
            double[] iv = inner.subInterval(i);
            if (inner.isInnerSlot(i)) {
                out.printf("%s├─[L%d slot %d] %s(fanout=%d) range=[%.0f,%.0f)%n",
                           prefix, depth, i, label, child.capacity(), iv[0], iv[1]);
                dumpInner(out, child, depth + 1, prefix + "│  ");
            } else {
                Leaf leaf = (Leaf) child;
                String name = (leaf instanceof DataNode) ? "DataNode"
                            : leaf.leafType().toString();
                out.printf("%s├─[L%d slot %d] %s(cap=%d, size=%d) range=[%.0f,%.0f)%n",
                           prefix, depth, i, name, leaf.capacity(), leaf.size(), iv[0], iv[1]);
            }
        }
    }

    // ── Types ──

    public static class IntEntry {
        public final double key;
        public final int bundleId;

        public IntEntry(double key, int bundleId) {
            this.key = key;
            this.bundleId = bundleId;
        }
    }

    public interface IntEntryVisitor {
        void visit(double key, int bundleId);
    }
}
