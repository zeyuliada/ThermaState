/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import thermastate.index.Leaf;
import thermastate.index.Node;

/**
 * Drift Detector — monitors per-interval-root drift and signals global rebuild.
 *
 * Formula (17): u_t = |{ region roots that need rebuild }|
 *   g_t = u_t / m
 *
 * Formula (18): trigger global rebuild if g_t ≥ τ_g for P consecutive checks.
 *
 * A region root "needs rebuild" when at least one of its descendant leaves
 * has an ActivityTracker score ≥ θ_l (appears in the activityCandidates set).
 *
 * Parameters:
 *   τ_g = 0.4  — global rebuild threshold
 *   P   = 3    — consecutive check periods required
 */
public final class DriftDetector {

    public static final double TAU_G = 0.4;
    public static final int P_CHECKS = 3;

    /** region root → all descendant leaves in that subtree */
    private final Map<Node, Set<Leaf>> regionLeaves;

    private int consecutiveCheckCount;
    private int needRebuildCount;
    private double driftScore;

    public DriftDetector() {
        this.regionLeaves = new LinkedHashMap<>();
        this.consecutiveCheckCount = 0;
        this.needRebuildCount = 0;
        this.driftScore = 0.0;
    }

    /** Register a region root and all leaves in its subtree. */
    public void registerRegion(Node regionRoot, Set<Leaf> leaves) {
        if (regionRoot == null) throw new NullPointerException("regionRoot");
        if (leaves == null) throw new NullPointerException("leaves");
        regionLeaves.put(regionRoot, leaves);
    }

    /** Total number of registered regions. */
    public int totalRegions() {
        return regionLeaves.size();
    }

    /**
     * Update drift statistics with current ActivityTracker rebuild candidates.
     * Call once per monitoring cycle (every ~2 seconds).
     *
     * @param activityCandidates  set of leaves whose activity ≥ θ_l
     * @return u_t — number of region roots that need rebuild
     */
    public int update(Set<Leaf> activityCandidates) {
        if (activityCandidates == null || activityCandidates.isEmpty()) {
            needRebuildCount = 0;
        } else {
            needRebuildCount = 0;
            for (Set<Leaf> leaves : regionLeaves.values()) {
                for (Leaf candidate : activityCandidates) {
                    if (leaves.contains(candidate)) {
                        needRebuildCount++;
                        break; // count each region at most once
                    }
                }
            }
        }

        int m = regionLeaves.size();
        driftScore = (m == 0) ? 0.0 : (double) needRebuildCount / m;

        if (driftScore >= TAU_G) {
            consecutiveCheckCount++;
        } else {
            consecutiveCheckCount = 0;
        }

        return needRebuildCount;
    }

    /** True when g_t ≥ τ_g has held for P consecutive checks (formula 18). */
    public boolean shouldGlobalRebuild() {
        return consecutiveCheckCount >= P_CHECKS;
    }

    /** Current drift score g_t = u_t / m. */
    public double getDriftScore() {
        return driftScore;
    }

    /** Number of region roots that currently need rebuild. */
    public int getNeedRebuildCount() {
        return needRebuildCount;
    }

    /** Consecutive checks with g_t ≥ τ_g. */
    public int getConsecutiveCheckCount() {
        return consecutiveCheckCount;
    }

    /** Reset all state (call after global rebuild completes). */
    public void reset() {
        consecutiveCheckCount = 0;
        needRebuildCount = 0;
        driftScore = 0.0;
        regionLeaves.clear();
    }
}
