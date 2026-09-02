/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import thermastate.agent.LeafAgent;
import thermastate.index.Index;
import thermastate.index.InnerNode;
import thermastate.index.Leaf;
import thermastate.index.LeafType;
import thermastate.index.Node;
import thermastate.monitor.LeafStats;
import thermastate.monitor.MonitorSummary;
import thermastate.monitor.TemperatureMonitor;

/**
 * Background reorganizer — bridges monitoring signals to rebuild actions.
 */
public final class Reorganizer implements Runnable {

    private static final Logger LOG = LoggerFactory.getLogger(Reorganizer.class);
    private static final long DEFAULT_INTERVAL_MS = 2000;
    private static final int PDF_SIZE = 1024;

    private final Index index;
    private final TemperatureMonitor monitor;
    private final ActivityTracker activityTracker;
    private final IntervalActivityTracker intervalActivityTracker = new IntervalActivityTracker();
    private final DriftDetector driftDetector;
    private final IntervalLock intervalLock;
    private final long intervalMs;

    private LeafAgent leafAgent;

    private volatile boolean running;
    private Thread thread;

    private final AtomicLong scanCount = new AtomicLong();
    private final AtomicLong rebuildCount = new AtomicLong();
    private final AtomicLong agentDecisionCount = new AtomicLong();
    private final AtomicLong heuristicFallbackCount = new AtomicLong();
    private volatile int lastCandidateCount;
    private volatile int lastIntervalCandidateCount;
    private volatile double lastDriftScore;
    private volatile boolean globalRebuildSignaled;

    private volatile Map<Leaf, LeafStats> lastLeafStats;

    public Reorganizer(Index index, TemperatureMonitor monitor,
                       ActivityTracker activityTracker, DriftDetector driftDetector,
                       IntervalLock intervalLock) {
        this(index, monitor, activityTracker, driftDetector, intervalLock, DEFAULT_INTERVAL_MS);
    }

    public Reorganizer(Index index, TemperatureMonitor monitor,
                       ActivityTracker activityTracker, DriftDetector driftDetector,
                       IntervalLock intervalLock, long intervalMs) {
        this.index = index;
        this.monitor = monitor;
        this.activityTracker = activityTracker;
        this.driftDetector = driftDetector;
        this.intervalLock = intervalLock;
        this.intervalMs = intervalMs;
        index.setIntervalLock(intervalLock);
        index.setDriftDetector(driftDetector);
    }

    public void setLeafAgentModel(String modelPath) {
        try {
            this.leafAgent = LeafAgent.load(modelPath);
            LOG.info("Reorganizer: LeafAgent loaded from {}", modelPath);
        } catch (Exception e) {
            LOG.warn("Reorganizer: failed to load LeafAgent from {} — "
                     + "falling back to heuristic. Cause: {}",
                     modelPath, e.getMessage());
            this.leafAgent = null;
        }
    }

    public boolean hasLeafAgent() { return leafAgent != null; }
    public long getAgentDecisionCount() { return agentDecisionCount.get(); }
    public long getHeuristicFallbackCount() { return heuristicFallbackCount.get(); }

    public synchronized void start() {
        if (running) return;
        running = true;
        thread = new Thread(this, "thermastate-reorganizer");
        thread.setDaemon(true);
        thread.start();
        LOG.info("Reorganizer started (interval={}ms, agent={})",
                 intervalMs, leafAgent != null ? "RL" : "heuristic");
    }

    public synchronized void stop() {
        running = false;
        if (thread != null) {
            thread.interrupt();
            try {
                thread.join(5000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            thread = null;
        }
        if (leafAgent != null) {
            leafAgent.close();
            leafAgent = null;
        }
        LOG.info("Reorganizer stopped (scans={} rebuilds={} agentDecisions={} heuristic={})",
                 scanCount.get(), rebuildCount.get(),
                 agentDecisionCount.get(), heuristicFallbackCount.get());
    }

    @Override
    public void run() {
        while (running) {
            try {
                Thread.sleep(intervalMs);
                if (!running) break;
                scan();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    void scan() {
        scanCount.incrementAndGet();

        MonitorSummary summary = monitor.getSummary();
        if (summary == null || summary.totalOps() == 0) {
            return;
        }

        Map<Leaf, LeafStats> leafStats = new HashMap<>();
        for (LeafStats s : summary.leafStats()) {
            leafStats.put(s.node(), s);
        }

        activityTracker.update(leafStats, summary.totalOps());
        Set<Leaf> candidates = activityTracker.getCandidates();
        lastCandidateCount = candidates.size();
        lastLeafStats = leafStats;

        // Aggregate per-leaf access to interval roots (leaf's direct parent routing node)
        Map<Node, Long> intervalOps = new HashMap<>();
        for (Map.Entry<Leaf, LeafStats> e : leafStats.entrySet()) {
            Index.ParentRef ref = index.findParent(e.getKey());
            if (ref == null) continue;
            Node intervalRoot = ref.parent;
            long ops = e.getValue().readCount() + e.getValue().writeCount();
            intervalOps.merge(intervalRoot, ops, Long::sum);
        }
        intervalActivityTracker.update(intervalOps, summary.totalOps());
        Set<Node> candidateIntervals = new HashSet<>(intervalActivityTracker.getCandidates());
        // Merge size-driven forced rebuilds (single-node ceiling overflow)
        candidateIntervals.addAll(index.drainForcedRebuilds());
        lastIntervalCandidateCount = candidateIntervals.size();

        int u_t = driftDetector.update(candidates);
        lastDriftScore = driftDetector.getDriftScore();

        LOG.debug("scan #{}: candidates={} intervals={} drift={} u_t={}",
                  scanCount.get(), lastCandidateCount, lastIntervalCandidateCount,
                  String.format("%.3f", lastDriftScore), u_t);

        if (!candidateIntervals.isEmpty()) {
            executeSubtreeRebuilds(candidateIntervals, leafStats);
        }

        if (driftDetector.shouldGlobalRebuild()) {
            globalRebuildSignaled = true;
            LOG.info("Global rebuild triggered: g_t={} u_t={}/{} checks={}/{}",
                     String.format("%.3f", lastDriftScore),
                     u_t, driftDetector.totalRegions(),
                     driftDetector.getConsecutiveCheckCount(), DriftDetector.P_CHECKS);
            try {
                // Reset before rebuild: bulkLoad re-registers drift regions via
                // Index.registerDriftRegions (wired through setDriftDetector).
                driftDetector.reset();
                if (GlobalRebuilder.rebuild(index)) {
                    LOG.info("Global rebuild complete");
                }
            } catch (Exception e) {
                LOG.warn("Global rebuild failed: {}", e.getMessage());
            }
        }
    }

    private void executeLeafRebuilds(Set<Leaf> candidates,
                                      Map<Leaf, LeafStats> stats) {
        for (Leaf leaf : candidates) {
            try {
                LeafStats s = stats.get(leaf);
                if (s == null) continue;

                LeafType newType;
                boolean usedAgent = false;

                if (leafAgent != null) {
                    newType = decideWithAgent(leaf, s);
                    usedAgent = true;
                } else {
                    newType = SubtreeRebuilder.chooseType(
                        leaf, s.fr(), s.fw(), s.dataSize());
                    heuristicFallbackCount.incrementAndGet();
                }

                if (newType == null || newType == leaf.leafType()) {
                    continue;
                }

                if (usedAgent) agentDecisionCount.incrementAndGet();

                Index.ParentRef ref = index.findParent(leaf);
                if (ref == null) continue;

                if (leaf.leafType() == LeafType.HOT_WRITE) {
                    rebuildHotWriteWithDelta(leaf, newType, ref, usedAgent, s);
                } else {
                    rebuildWithWriteLock(leaf, newType, ref, usedAgent, s);
                }

                activityTracker.reset(leaf);

            } catch (Exception e) {
                LOG.warn("Leaf rebuild failed for {}: {}", leaf, e.getMessage());
            }
        }
    }

    /**
     * Subtree reconstruction (paper stage 2): rebuild each candidate interval
     * root in the background and atomically swap it into place. First version
     * holds the interval write lock for the whole rebuild — correctness over
     * zero-lock concurrency; the delta-log optimization can follow later.
     */
    private void executeSubtreeRebuilds(Set<Node> candidateIntervals,
                                        Map<Leaf, LeafStats> stats) {
        for (Node intervalRoot : candidateIntervals) {
            if (!(intervalRoot instanceof InnerNode)) continue;
            try {
                Index.ParentRef ref = index.findParent(intervalRoot);
                if (ref == null) continue;

                intervalLock.writeLock(ref.parent);
                try {
                    Node current = index.childAt(ref);
                    if (current != intervalRoot) continue;

                    Node newSubtree = index.rebuildSubtree(intervalRoot);
                    index.replaceChild(ref.parent, ref.position, newSubtree);
                    rebuildCount.incrementAndGet();
                    LOG.info("Subtree rebuilt: interval=[{},{}] fanout {}->{}",
                             intervalRoot.lower(), intervalRoot.upper(),
                             intervalRoot.capacity(), newSubtree.capacity());
                } finally {
                    intervalLock.writeUnlock(ref.parent);
                }
                intervalActivityTracker.reset(intervalRoot);
            } catch (Exception e) {
                LOG.warn("Subtree rebuild failed for {}: {}", intervalRoot, e.getMessage());
            }
        }
    }

    /** Original single-writeLock rebuild for ColdLeaf / HotReadLeaf / DataNode. */
    private void rebuildWithWriteLock(Leaf leaf, LeafType newType,
                                       Index.ParentRef ref,
                                       boolean usedAgent, LeafStats s) {
        intervalLock.writeLock(ref.parent);
        try {
            Leaf newLeaf = SubtreeRebuilder.rebuildLeaf(leaf, newType);
            index.replaceChild(ref.parent, ref.position, newLeaf);
            rebuildCount.incrementAndGet();
            LOG.info("Leaf rebuilt [{}]: {} → {} @ range=[%.0f,%.0f] fr={:.3f} fw={:.3f}",
                     usedAgent ? "RL" : "heuristic",
                     leaf.leafType(), newType,
                     leaf.lower(), leaf.upper(), s.fr(), s.fw());
        } finally {
            intervalLock.writeUnlock(ref.parent);
        }
    }

    /**
     * Three-phase delta-log rebuild for HotWriteLeaf.
     *
     * Phase 1: writeLock → set pendingDelta → writeUnlock  (∼100 ns)
     * Phase 2: rebuild new leaf from snapshot (no lock, ∼1 ms)
     * Phase 3: writeLock → verify → replay delta → swap → writeUnlock (∼100 ns + replay)
     *
     * Concurrent writes during Phase 2 are redirected to the delta buffer
     * and merged in Phase 3, so the write path is never blocked.
     */
    private void rebuildHotWriteWithDelta(Leaf oldLeaf, LeafType newType,
                                           Index.ParentRef ref,
                                           boolean usedAgent, LeafStats s) {
        Leaf.DeltaBuffer delta = new Leaf.DeltaBuffer();

        // Phase 1 — mark (exclusive, minimal)
        intervalLock.writeLock(ref.parent);
        try {
            oldLeaf.pendingDelta = delta;
        } finally {
            intervalLock.writeUnlock(ref.parent);
        }

        try {
            // Phase 2 — rebuild (no lock, writes go to delta)
            Leaf newLeaf = SubtreeRebuilder.rebuildLeaf(oldLeaf, newType);

            // Phase 3 — merge + swap (exclusive, minimal)
            intervalLock.writeLock(ref.parent);
            try {
                Leaf current = index.routeToLeafForRef(ref);
                if (current != oldLeaf) {
                    // Leaf was expanded or replaced during rebuild —
                    // replay delta onto the current leaf instead.
                    delta.replay(current);
                    LOG.info("Leaf rebuilt [{}]: {} → {} @ range=[%.0f,%.0f] "
                             + "(leaf replaced during rebuild, replayed {} delta ops)",
                             usedAgent ? "RL" : "heuristic",
                             oldLeaf.leafType(), current.leafType(),
                             oldLeaf.lower(), oldLeaf.upper(), delta.opCount());
                } else {
                    delta.replay(newLeaf);
                    index.replaceChild(ref.parent, ref.position, newLeaf);
                    rebuildCount.incrementAndGet();
                    LOG.info("Leaf rebuilt [{}]: {} → {} @ range=[%.0f,%.0f] "
                             + "fr={:.3f} fw={:.3f} (delta ops={})",
                             usedAgent ? "RL" : "heuristic",
                             oldLeaf.leafType(), newType,
                             oldLeaf.lower(), oldLeaf.upper(),
                             s.fr(), s.fw(), delta.opCount());
                }
            } finally {
                intervalLock.writeUnlock(ref.parent);
            }
        } finally {
            // Always clear pendingDelta — old leaf is retiring
            oldLeaf.pendingDelta = null;
        }
    }

    private LeafType decideWithAgent(Leaf leaf, LeafStats s) {
        try {
            float[] pdf = computeLeafPDF(leaf);
            if (pdf == null) {
                return SubtreeRebuilder.chooseType(leaf, s.fr(), s.fw(), s.dataSize());
            }

            LeafAgent.Decision d = leafAgent.decide(
                pdf, leaf.size(), (float) s.fr(), (float) s.fw());

            if (d.isLeaf && d.leafType != null) {
                return d.leafType;
            }
            LOG.debug("LeafAgent recommended expand(fanout={}) for leaf [%.0f,%.0f]; "
                      + "falling back to heuristic",
                      d.fanout, leaf.lower(), leaf.upper());
            return SubtreeRebuilder.chooseType(leaf, s.fr(), s.fw(), s.dataSize());

        } catch (Exception e) {
            LOG.warn("LeafAgent inference failed for leaf [%.0f,%.0f]: {} — "
                     + "falling back to heuristic",
                     leaf.lower(), leaf.upper(), e.getMessage());
            heuristicFallbackCount.incrementAndGet();
            return SubtreeRebuilder.chooseType(leaf, s.fr(), s.fw(), s.dataSize());
        }
    }

    private static float[] computeLeafPDF(Leaf leaf) {
        double lo = leaf.lower();
        double hi = leaf.upper();
        if (hi <= lo) return null;

        int n = leaf.size();
        if (n == 0) return null;

        float[] pdf = new float[PDF_SIZE];
        double range = hi - lo;

        leaf.forEachEntry((key, bundleId) -> {
            int bin = (int) ((key - lo) / range * PDF_SIZE);
            if (bin < 0) bin = 0;
            if (bin >= PDF_SIZE) bin = PDF_SIZE - 1;
            pdf[bin] += 1.0f / n;
        });

        return pdf;
    }

    public long getScanCount() { return scanCount.get(); }
    public long getRebuildCount() { return rebuildCount.get(); }
    public boolean isRunning() { return running; }
    public int getLastCandidateCount() { return lastCandidateCount; }
    public int getLastIntervalCandidateCount() { return lastIntervalCandidateCount; }
    public double getLastDriftScore() { return lastDriftScore; }
    public boolean globalRebuildSignaled() { return globalRebuildSignaled; }

    public void forceScan() {
        scan();
    }
}
