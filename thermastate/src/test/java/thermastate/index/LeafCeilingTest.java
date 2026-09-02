/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

import static org.junit.Assert.*;

import java.util.Set;

import org.junit.Test;

import thermastate.reorganizer.IntervalLock;

public class LeafCeilingTest {

    @Test
    public void testHotWriteLeafTotalEntriesIncludesOverflow() {
        HotWriteLeaf leaf = HotWriteLeaf.create(16, 0, 100);
        for (int i = 0; i < 16; i++) leaf.bulkInsert(i, i); // array full
        assertEquals(16, leaf.size());
        assertEquals(16, leaf.totalEntries());

        // Out-of-order insert that forces an overflow chain entry
        leaf.put(5.5, 999);
        assertEquals(16, leaf.size());
        assertEquals(17, leaf.totalEntries());
    }

    @Test
    public void testDataNodeTotalEntriesEqualsSize() {
        DataNode leaf = new DataNode(32, 0, 100);
        for (int i = 0; i < 10; i++) leaf.put(i, i);
        assertEquals(10, leaf.totalEntries());
    }

    @Test
    public void testForceRebuildAndDrain() {
        Index idx = new Index(new Configuration(), 0, 100);
        InnerNode interval = InnerNode.create(2, 0, 100);

        assertTrue(idx.drainForcedRebuilds().isEmpty());

        idx.forceRebuild(interval);
        Set<Node> forced = idx.drainForcedRebuilds();
        assertEquals(1, forced.size());
        assertTrue(forced.contains(interval));

        // drain clears the set
        assertTrue(idx.drainForcedRebuilds().isEmpty());
    }

    @Test
    public void testPutFlagsForceRebuildWhenExceedingCeiling() {
        Index idx = new Index(new Configuration(), 0, 4000);

        InnerNode interval = InnerNode.create(1, 0, 4000);
        DataNode leaf = new DataNode(64, 0, 4000);
        interval.setLeafChild(0, leaf);
        idx.replaceRoot(interval);
        idx.setIntervalLock(new IntervalLock());

        // Insert past the single-node ceiling (MAX_LEAF_ENTRIES = 1024)
        for (int i = 0; i < 1100; i++) {
            idx.put(i * 1.0, i);
        }

        Set<Node> forced = idx.drainForcedRebuilds();
        assertFalse("force rebuild should be flagged once the leaf exceeds the ceiling",
                    forced.isEmpty());
    }
}
