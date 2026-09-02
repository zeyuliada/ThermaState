/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.monitor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import thermastate.index.InnerNode;
import thermastate.index.Leaf;

/**
 * Sliding-window temperature monitor for ThermaState index.
 *
 * Uses time-sliced buckets: the window (default 10s) is divided into
 * 10 slices of 1s each.  record*() adds to the current slice; when a
 * slice expires it is overwritten.  Query methods merge all live slices
 * and compute temperature scores on the fly.
 *
 * Architecture:
 *   InnerNode (level &lt; h): heuristic label via T_route ≥ τ
 *   InnerNode (level ≥ h): raw routeCount only, no label
 *   Leaf:               raw readCount/writeCount, T_read/T_write, no label
 *
 * Formulas (paper eq. 6-9):
 *   A_ij   = log(1 + freq_ij) / log(1 + freq_max)
 *   R_ij   = (u_ij - l_ij) / (K_max - K_min)
 *   T_ij   = A_ij * (1 + β * R_ij)
 *   label  = T_ij ≥ τ ? HOT : COLD  (only when level &lt; h)
 */
public class TemperatureMonitor {

    static final double WINDOW_SECONDS = 10.0;
    static final int    SLICE_COUNT    = 10;
    static final double BETA = 0.5;
    static final double TAU  = 0.3;

    private final int h;
    private final double kMin;
    private final double kMax;
    private final double kRange;
    private final double windowSeconds;
    private final double sliceSeconds;

    // Ring buffer of time slices
    private final TimeSlice[] slices;
    private int head;
    private long currentSliceStartNs;

    /** Single time-slice bucket (1s by default). */
    private static class TimeSlice {
        long startNs;
        long endNs;
        long totalOps;
        final Map<InnerNode, long[]> innerRoutes = new HashMap<>();  // node → [routeCount]
        final Map<InnerNode, Integer> innerLevels = new HashMap<>();  // node → level
        final Map<Leaf, long[]> leafOps = new HashMap<>();        // node → [reads, writes]

        void reset(long startNs) {
            this.startNs = startNs;
            this.endNs = 0;
            this.totalOps = 0;
            this.innerRoutes.clear();
            this.innerLevels.clear();
            this.leafOps.clear();
        }
    }

    public TemperatureMonitor(int h, double kMin, double kMax) {
        this(h, kMin, kMax, WINDOW_SECONDS, SLICE_COUNT);
    }

    public TemperatureMonitor(int h, double kMin, double kMax,
                              double windowSeconds, int sliceCount) {
        this.h = h;
        this.kMin = kMin;
        this.kMax = kMax;
        this.kRange = kMax - kMin;
        this.windowSeconds = windowSeconds;
        this.sliceSeconds = windowSeconds / sliceCount;
        this.slices = new TimeSlice[sliceCount];
        for (int i = 0; i < sliceCount; i++) {
            this.slices[i] = new TimeSlice();
        }
        this.head = 0;
        this.currentSliceStartNs = System.nanoTime();
        this.slices[head].reset(currentSliceStartNs);
    }

    // ── Recording API (called from Index.get/put) ──

    public void recordRoute(InnerNode node, int level) {
        rotateIfNeeded();
        TimeSlice s = slices[head];
        long[] c = s.innerRoutes.computeIfAbsent(node, k -> new long[1]);
        c[0]++;
        s.innerLevels.putIfAbsent(node, level);
        s.totalOps++;
    }

    public void recordRead(Leaf node) {
        rotateIfNeeded();
        TimeSlice s = slices[head];
        long[] c = s.leafOps.computeIfAbsent(node, k -> new long[2]);
        c[0]++;
        s.totalOps++;
    }

    public void recordWrite(Leaf node) {
        rotateIfNeeded();
        TimeSlice s = slices[head];
        long[] c = s.leafOps.computeIfAbsent(node, k -> new long[2]);
        c[1]++;
        s.totalOps++;
    }

    // ── Query API (always computed from live slices) ──

    public List<InnerNodeStats> getInnerStats() {
        MonitorSummary sum = buildSummary();
        return sum != null ? sum.innerStats() : Collections.emptyList();
    }

    public List<InnerNodeStats> getHotInnerNodes() {
        List<InnerNodeStats> result = new ArrayList<>();
        MonitorSummary sum = buildSummary();
        if (sum == null) return result;
        for (InnerNodeStats s : sum.innerStats()) {
            if (s.isHot()) result.add(s);
        }
        result.sort((a, b) -> Double.compare(b.T_route, a.T_route));
        return result;
    }

    public List<InnerNodeStats> getColdInnerNodes() {
        List<InnerNodeStats> result = new ArrayList<>();
        MonitorSummary sum = buildSummary();
        if (sum == null) return result;
        for (InnerNodeStats s : sum.innerStats()) {
            if (s.isCold()) result.add(s);
        }
        return result;
    }

    public List<LeafStats> getLeafStats() {
        MonitorSummary sum = buildSummary();
        return sum != null ? sum.leafStats() : Collections.emptyList();
    }

    public MonitorSummary getSummary() {
        return buildSummary();
    }

    // ── Slice rotation ──

    private void rotateIfNeeded() {
        long now = System.nanoTime();
        double elapsed = (now - currentSliceStartNs) / 1_000_000_000.0;
        if (elapsed < sliceSeconds) return;

        // Seal current slice
        slices[head].endNs = now;

        // Advance head, overwriting the oldest slice
        head = (head + 1) % slices.length;
        currentSliceStartNs = now;
        slices[head].reset(now);
    }

    // ── Snapshot builder ──

    private MonitorSummary buildSummary() {
        long now = System.nanoTime();
        long cutoffNs = now - (long)(windowSeconds * 1_000_000_000.0);

        // Merge all slices that overlap the window
        Map<InnerNode, long[]> mergedInner = new HashMap<>();
        Map<InnerNode, Integer> innerLevels = new HashMap<>();
        Map<Leaf, long[]> mergedLeaf = new HashMap<>();
        long totalOps = 0;

        for (TimeSlice s : slices) {
            if (s.endNs == 0) {
                // Current (unsealed) slice — always include
            } else if (s.endNs < cutoffNs) {
                continue; // expired
            }
            if (s.totalOps == 0) continue;

            totalOps += s.totalOps;

            for (Map.Entry<InnerNode, long[]> e : s.innerRoutes.entrySet()) {
                long[] acc = mergedInner.computeIfAbsent(e.getKey(), k -> new long[1]);
                acc[0] += e.getValue()[0];
                innerLevels.putIfAbsent(e.getKey(), s.innerLevels.get(e.getKey()));
            }
            for (Map.Entry<Leaf, long[]> e : s.leafOps.entrySet()) {
                long[] acc = mergedLeaf.computeIfAbsent(e.getKey(), k -> new long[2]);
                long[] v = e.getValue();
                acc[0] += v[0];
                acc[1] += v[1];
            }
        }

        if (totalOps == 0) return null;

        // Max counts for normalization
        long maxInnerRoute = 0;
        for (long[] v : mergedInner.values()) {
            if (v[0] > maxInnerRoute) maxInnerRoute = v[0];
        }
        long maxLeafRead = 0, maxLeafWrite = 0;
        for (long[] v : mergedLeaf.values()) {
            if (v[0] > maxLeafRead)  maxLeafRead  = v[0];
            if (v[1] > maxLeafWrite) maxLeafWrite = v[1];
        }

        double denomInner = Math.log(1 + maxInnerRoute);
        double denomRead  = Math.log(1 + maxLeafRead);
        double denomWrite = Math.log(1 + maxLeafWrite);

        // Build InnerNode stats
        List<InnerNodeStats> innerList = new ArrayList<>();
        for (Map.Entry<InnerNode, long[]> e : mergedInner.entrySet()) {
            InnerNode node = e.getKey();
            long rc = e.getValue()[0];
            Integer level = innerLevels.get(node);
            if (level == null) level = 0;

            InnerNodeStats stats = new InnerNodeStats(node, level);
            stats.routeCount = rc;
            double A = denomInner > 0 ? Math.log(1 + rc) / denomInner : 0.0;
            double R = kRange > 0 ? (node.upper() - node.lower()) / kRange : 0.0;
            stats.T_route = A * (1.0 + BETA * R);
            if (level < h) {
                stats.label = stats.T_route >= TAU ? InnerNodeStats.Label.HOT
                                                   : InnerNodeStats.Label.COLD;
            }
            innerList.add(stats);
        }

        // Build Leaf stats
        List<LeafStats> leafList = new ArrayList<>();
        for (Map.Entry<Leaf, long[]> e : mergedLeaf.entrySet()) {
            Leaf node = e.getKey();
            long[] v = e.getValue();

            LeafStats stats = new LeafStats(node);
            stats.readCount  = v[0];
            stats.writeCount = v[1];
            double A_r = denomRead > 0
                    ? Math.log(1 + v[0]) / denomRead : 0.0;
            double A_w = denomWrite > 0
                    ? Math.log(1 + v[1]) / denomWrite : 0.0;
            double R = kRange > 0 ? (node.upper() - node.lower()) / kRange : 0.0;
            stats.T_read  = A_r * (1.0 + BETA * R);
            stats.T_write = A_w * (1.0 + BETA * R);
            stats.fr = totalOps > 0 ? (double) v[0] / totalOps : 0.0;
            stats.fw = totalOps > 0 ? (double) v[1] / totalOps : 0.0;
            leafList.add(stats);
        }

        return new MonitorSummary(0, totalOps, innerList, leafList);
    }

    /** Force-expire all current slices and start fresh (for testing). */
    public void forceWindowClose() {
        long now = System.nanoTime();
        for (TimeSlice s : slices) {
            s.endNs = now;
            s.totalOps = 0;
            s.innerRoutes.clear();
            s.innerLevels.clear();
            s.leafOps.clear();
        }
        head = 0;
        currentSliceStartNs = now;
        slices[head].reset(now);
    }
}
