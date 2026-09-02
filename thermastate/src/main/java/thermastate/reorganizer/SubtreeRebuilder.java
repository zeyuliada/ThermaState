/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.ArrayList;
import java.util.List;

import thermastate.index.Index;
import thermastate.index.Leaf;
import thermastate.index.LeafType;
import thermastate.index.Node;

/**
 * Copy-on-Rebuild for individual leaves.
 */
public final class SubtreeRebuilder {

    private SubtreeRebuilder() {}

    public static final class Snapshot {
        public final double key;
        public final int bundleId;

        Snapshot(double key, int bundleId) {
            this.key = key;
            this.bundleId = bundleId;
        }
    }

    /** Rebuild a single leaf with a new type, preserving all data. */
    public static Leaf rebuildLeaf(Leaf oldLeaf, LeafType newType) {
        List<Snapshot> entries = snapshotLeaf(oldLeaf);
        entries.sort((a, b) -> Double.compare(a.key, b.key));

        Leaf newLeaf = Index.createLeaf(newType, oldLeaf.capacity(),
                                         oldLeaf.lower(), oldLeaf.upper());

        for (Snapshot e : entries) {
            Index.bulkInsertInto(newLeaf, e.key, e.bundleId);
        }

        return newLeaf;
    }

    /** Heuristic leaf-type selection. Fallback when LeafAgent model is not available. */
    public static LeafType chooseType(Leaf current, double fr, double fw, int dataSize) {
        LeafType cur = current.leafType();

        if (dataSize <= 8) {
            return cur == LeafType.COLD ? cur : LeafType.COLD;
        }

        if (fr > 0.6 && cur != LeafType.HOT_READ) {
            return LeafType.HOT_READ;
        }

        if (fw > 0.4 && cur != LeafType.HOT_WRITE) {
            return LeafType.HOT_WRITE;
        }

        if (fr < 0.15 && fw < 0.15 && cur != LeafType.COLD) {
            return LeafType.COLD;
        }

        return cur;
    }

    private static List<Snapshot> snapshotLeaf(Leaf leaf) {
        List<Snapshot> out = new ArrayList<>(leaf.size());
        leaf.forEachEntry((key, bundleId) -> out.add(new Snapshot(key, bundleId)));
        return out;
    }

    /** Snapshot all entries from a subtree root. */
    public static List<Snapshot> snapshotSubtree(Node root) {
        List<Snapshot> out = new ArrayList<>();
        collectSubtreeEntries(root, out);
        return out;
    }

    private static void collectSubtreeEntries(Node node, List<Snapshot> out) {
        if (node instanceof thermastate.index.InnerNode) {
            thermastate.index.InnerNode inner = (thermastate.index.InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) {
                Node child = inner.getChild(i);
                if (child != null) collectSubtreeEntries(child, out);
            }
        } else if (node instanceof thermastate.index.ColdInnerNode) {
            thermastate.index.ColdInnerNode cold = (thermastate.index.ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) {
                Node child = cold.getChild(i);
                if (child != null) collectSubtreeEntries(child, out);
            }
        } else if (node instanceof Leaf) {
            Leaf leaf = (Leaf) node;
            leaf.forEachEntry((key, bundleId) -> out.add(new Snapshot(key, bundleId)));
        }
    }
}
