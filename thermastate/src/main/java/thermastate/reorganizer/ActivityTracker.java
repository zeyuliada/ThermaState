/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import thermastate.index.Leaf;
import thermastate.monitor.LeafStats;

/**
 * Tracks per-leaf activity scores (formula 13).
 *
 * a_i(t) = max(0, β · a_i(t-1) + x_i(t) − τ_l)     ... (13)
 *   where x_i = (readCount + writeCount) / total_window_ops (normalized)
 *
 * A leaf becomes a rebuild candidate when a_i(t) ≥ τ_l.
 * τ_l serves as both the low-pass noise filter and the trigger threshold.
 *
 * Called by the Reorganizer at each window close.
 */
public final class ActivityTracker {

    /** Decay factor β — how fast old activity fades (formula 13). */
    public static final double BETA = 0.85;

    /** Trigger threshold τ_l — noise filter and rebuild trigger (formula 13). */
    public static final double TAU_L = 0.2;

    private final Map<Leaf, Double> scores = new ConcurrentHashMap<>();
    private final Set<Leaf> candidates = ConcurrentHashMap.newKeySet();

    /**
     * Update activity scores from the latest window stats.
     * @param stats   per-leaf stats from TemperatureMonitor
     * @param totalWindowOps  total read+write operations across all leaves this window
     */
    public void update(Map<Leaf, LeafStats> stats, long totalWindowOps) {
        // Decay all existing scores
        for (Leaf leaf : scores.keySet()) {
            scores.computeIfPresent(leaf, (k, v) -> v * BETA);
        }

        double effectiveTotal = Math.max(1.0, (double) totalWindowOps);

        // Accumulate this window's activity
        for (Map.Entry<Leaf, LeafStats> e : stats.entrySet()) {
            Leaf leaf = e.getKey();
            LeafStats s = e.getValue();
            long ops = s.readCount() + s.writeCount();
            double xi = ops / effectiveTotal;           // normalized [0,1]
            double prev = scores.getOrDefault(leaf, 0.0);
            double next = Math.max(0.0, prev + xi - TAU_L);
            scores.put(leaf, next);

            if (next >= TAU_L) {
                candidates.add(leaf);
            } else {
                candidates.remove(leaf);
            }
        }
    }

    /** Get the set of leaves whose activity score has crossed θ_l. */
    public Set<Leaf> getCandidates() {
        return Set.copyOf(candidates);
    }

    /** Get the current activity score for a leaf (0 if not tracked). */
    public double getScore(Leaf leaf) {
        return scores.getOrDefault(leaf, 0.0);
    }

    /** Reset a leaf's activity score after successful rebuild. */
    public void reset(Leaf leaf) {
        scores.remove(leaf);
        candidates.remove(leaf);
    }

    /** Number of tracked leaves. */
    public int trackedCount() {
        return scores.size();
    }
}
