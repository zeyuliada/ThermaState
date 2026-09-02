/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import thermastate.index.Node;

/**
 * Tracks per-interval-root activity scores (paper formula 13).
 *
 * a_i(t) = max(0, β · a_i(t-1) + x_i(t) − τ_l)     ... (13)
 *   where x_i = interval read+write ops / total_window_ops (normalized)
 *
 * An interval root (a leaf's direct parent routing node) becomes a subtree
 * rebuild candidate when a_i(t) ≥ τ_l. Unlike {@link ActivityTracker}, which
 * tracks per-leaf activity, this tracks per-interval-root activity so the
 * rebuild unit matches the subtree-reconstruction granularity in the paper.
 */
public final class IntervalActivityTracker {

    /** Decay factor β — how fast old activity fades (formula 13). */
    public static final double BETA = 0.85;

    /** Trigger threshold τ_l — noise filter and subtree-rebuild trigger. */
    public static final double TAU_L = 0.2;

    private final Map<Node, Double> scores = new ConcurrentHashMap<>();
    private final Set<Node> candidates = ConcurrentHashMap.newKeySet();

    /**
     * Update activity scores from the latest window's per-interval access counts.
     *
     * @param intervalOps    interval root → total read+write ops this window
     * @param totalWindowOps total read+write ops across all intervals this window
     */
    public void update(Map<Node, Long> intervalOps, long totalWindowOps) {
        for (Node root : scores.keySet()) {
            scores.computeIfPresent(root, (k, v) -> v * BETA);
        }

        double effectiveTotal = Math.max(1.0, (double) totalWindowOps);

        for (Map.Entry<Node, Long> e : intervalOps.entrySet()) {
            Node root = e.getKey();
            double xi = e.getValue() / effectiveTotal;
            double prev = scores.getOrDefault(root, 0.0);
            double next = Math.max(0.0, prev + xi - TAU_L);
            scores.put(root, next);

            if (next >= TAU_L) {
                candidates.add(root);
            } else {
                candidates.remove(root);
            }
        }
    }

    /** Get the interval roots whose activity score has crossed τ_l. */
    public Set<Node> getCandidates() {
        return Set.copyOf(candidates);
    }

    /** Current activity score for an interval root (0 if not tracked). */
    public double getScore(Node root) {
        return scores.getOrDefault(root, 0.0);
    }

    /** Reset an interval root's activity score after successful rebuild. */
    public void reset(Node root) {
        scores.remove(root);
        candidates.remove(root);
    }

    /** Number of tracked interval roots. */
    public int trackedCount() {
        return scores.size();
    }
}
