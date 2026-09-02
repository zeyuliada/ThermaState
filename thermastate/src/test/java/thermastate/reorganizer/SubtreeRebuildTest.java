/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.junit.Test;

import thermastate.index.ColdInnerNode;
import thermastate.index.Configuration;
import thermastate.index.Index;
import thermastate.index.InnerNode;
import thermastate.index.Leaf;
import thermastate.index.LeafType;
import thermastate.index.Node;

public class SubtreeRebuildTest {

    private static List<Index.IntEntry> collect(Node node) {
        List<Index.IntEntry> out = new ArrayList<>();
        collectInto(node, out);
        out.sort(Comparator.comparingDouble(e -> e.key));
        return out;
    }

    private static void collectInto(Node node, List<Index.IntEntry> out) {
        if (node == null) return;
        if (node instanceof InnerNode) {
            InnerNode inner = (InnerNode) node;
            for (int i = 0; i < inner.capacity(); i++) collectInto(inner.getChild(i), out);
        } else if (node instanceof ColdInnerNode) {
            ColdInnerNode cold = (ColdInnerNode) node;
            for (int i = 0; i < cold.capacity(); i++) collectInto(cold.getChild(i), out);
        } else if (node instanceof Leaf) {
            Leaf leaf = (Leaf) node;
            leaf.forEachEntry((k, v) -> out.add(new Index.IntEntry(k, v)));
        }
    }

    /** Build an InnerNode interval with {@code fanout} cold leaves, each holding
     *  {@code entriesPerLeaf} uniformly-spaced keys. */
    private static InnerNode buildInterval(int fanout, int entriesPerLeaf,
                                           double lo, double hi) {
        InnerNode interval = InnerNode.create(fanout, lo, hi);
        double width = (hi - lo) / fanout;
        int key = 0;
        for (int slot = 0; slot < fanout; slot++) {
            double l = lo + slot * width;
            double h = l + width;
            Leaf leaf = Index.createLeaf(LeafType.COLD, entriesPerLeaf * 4, l, h);
            for (int j = 0; j < entriesPerLeaf; j++) {
                double k = l + (j + 0.5) * width / entriesPerLeaf;
                Index.bulkInsertInto(leaf, k, key++);
            }
            interval.setLeafChild(slot, leaf);
        }
        return interval;
    }

    @Test
    public void testRebuildPreservesData() {
        Index idx = new Index(new Configuration(), 0, 100);
        InnerNode interval = buildInterval(2, 20, 0, 100); // 40 entries

        List<Index.IntEntry> before = collect(interval);
        assertEquals(40, before.size());

        Node rebuilt = idx.rebuildSubtree(interval);

        List<Index.IntEntry> after = collect(rebuilt);
        assertEquals(before.size(), after.size());
        for (int i = 0; i < before.size(); i++) {
            assertEquals(before.get(i).key, after.get(i).key, 1e-9);
            assertEquals(before.get(i).bundleId, after.get(i).bundleId);
        }
    }

    @Test
    public void testRebuildSplitsWhenDataGrows() {
        Index idx = new Index(new Configuration(), 0, 100);
        // 2-slot interval holding 80 entries → rebuild should fan out to 4
        InnerNode interval = buildInterval(2, 40, 0, 100);

        Node rebuilt = idx.rebuildSubtree(interval);
        assertTrue(rebuilt instanceof InnerNode);
        assertEquals(4, rebuilt.capacity());
        assertEquals(80, collect(rebuilt).size());
    }

    @Test
    public void testRebuildDoesNotShrinkBelowCurrentFanout() {
        Index idx = new Index(new Configuration(), 0, 100);
        // 8-slot interval with 16 entries → must stay >= 8 fanout
        InnerNode interval = buildInterval(8, 2, 0, 100);

        Node rebuilt = idx.rebuildSubtree(interval);
        assertTrue(rebuilt.capacity() >= 8);
        assertEquals(16, collect(rebuilt).size());
    }

    @Test
    public void testRebuildEmptySubtreeReturnsOriginal() {
        Index idx = new Index(new Configuration(), 0, 100);
        InnerNode empty = InnerNode.create(4, 0, 100);
        Node rebuilt = idx.rebuildSubtree(empty);
        assertSame(empty, rebuilt);
    }
}
