/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import thermastate.index.InnerNode;
import thermastate.index.Node;

public class IntervalActivityTrackerTest {

    private static InnerNode interval(int fanout, double lo, double hi) {
        return InnerNode.create(fanout, lo, hi);
    }

    private static Map<Node, Long> ops(Node root, long count) {
        Map<Node, Long> m = new HashMap<>();
        m.put(root, count);
        return m;
    }

    @Test
    public void testCandidateWhenScoreExceedsThreshold() {
        IntervalActivityTracker t = new IntervalActivityTracker();
        InnerNode root = interval(4, 0, 100);

        // xi = 100/100 = 1.0 → a_i = 1.0 - 0.2 = 0.8 ≥ 0.2
        t.update(ops(root, 100), 100);

        assertEquals(1, t.getCandidates().size());
        assertTrue(t.getCandidates().contains(root));
        assertEquals(0.8, t.getScore(root), 1e-9);
    }

    @Test
    public void testNoCandidateWhenScoreBelowThreshold() {
        IntervalActivityTracker t = new IntervalActivityTracker();
        InnerNode root = interval(4, 0, 100);

        // xi = 10/100 = 0.1 → a_i = max(0, 0.1 - 0.2) = 0 < 0.2
        t.update(ops(root, 10), 100);

        assertEquals(0, t.getCandidates().size());
        assertEquals(0.0, t.getScore(root), 1e-9);
    }

    @Test
    public void testAccumulationAcrossWindows() {
        IntervalActivityTracker t = new IntervalActivityTracker();
        InnerNode root = interval(4, 0, 100);

        // window 1: xi=0.5 → a=0.3 ≥ 0.2 → candidate
        t.update(ops(root, 50), 100);
        assertTrue(t.getCandidates().contains(root));

        // window 2: decay 0.3*0.85=0.255, xi=0.5 → a=0.255+0.5-0.2=0.555
        t.update(ops(root, 50), 100);
        assertEquals(0.555, t.getScore(root), 1e-9);
    }

    @Test
    public void testResetClearsScoreAndCandidate() {
        IntervalActivityTracker t = new IntervalActivityTracker();
        InnerNode root = interval(4, 0, 100);

        t.update(ops(root, 100), 100);
        assertEquals(1, t.getCandidates().size());

        t.reset(root);
        assertEquals(0.0, t.getScore(root), 1e-9);
        assertEquals(0, t.getCandidates().size());
        assertEquals(0, t.trackedCount());
    }

    @Test
    public void testMultipleIntervalsTrackedIndependently() {
        IntervalActivityTracker t = new IntervalActivityTracker();
        InnerNode a = interval(2, 0, 50);
        InnerNode b = interval(2, 50, 100);

        Map<Node, Long> m = new HashMap<>();
        m.put(a, 90L);  // xi=0.9 → a=0.7
        m.put(b, 10L);  // xi=0.1 → a=0
        t.update(m, 100);

        assertEquals(0.7, t.getScore(a), 1e-9);
        assertEquals(0.0, t.getScore(b), 1e-9);
        assertEquals(1, t.getCandidates().size());
        assertTrue(t.getCandidates().contains(a));
        assertFalse(t.getCandidates().contains(b));
    }
}
