/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.monitor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Snapshot of a completed monitoring window.
 */
public class MonitorSummary {

    private final long windowId;
    private final long totalOps;
    private final List<InnerNodeStats> innerStats;
    private final List<LeafStats> leafStats;
    private final int hotInnerCount;
    private final int coldInnerCount;
    private final double hotRatio;

    MonitorSummary(long windowId, long totalOps,
                   List<InnerNodeStats> innerStats,
                   List<LeafStats> leafStats) {
        this.windowId = windowId;
        this.totalOps = totalOps;
        this.innerStats = Collections.unmodifiableList(innerStats);
        this.leafStats = Collections.unmodifiableList(leafStats);
        int hot = 0, cold = 0;
        for (InnerNodeStats s : innerStats) {
            if (s.isHot()) hot++;
            else if (s.isCold()) cold++;
        }
        this.hotInnerCount = hot;
        this.coldInnerCount = cold;
        int labeled = hot + cold;
        this.hotRatio = labeled > 0 ? (double) hot / labeled : 0.0;
    }

    public long windowId() { return windowId; }
    public long totalOps() { return totalOps; }
    public List<InnerNodeStats> innerStats() { return innerStats; }
    public List<LeafStats> leafStats() { return leafStats; }
    public int hotInnerCount() { return hotInnerCount; }
    public int coldInnerCount() { return coldInnerCount; }
    public int labeledInnerCount() { return hotInnerCount + coldInnerCount; }
    public double hotRatio() { return hotRatio; }

    @Override
    public String toString() {
        return String.format("MonitorSummary[win=%d ops=%d inner=%d(hot=%d cold=%d ratio=%.2f) leaves=%d]",
                windowId, totalOps, innerStats.size(),
                hotInnerCount, coldInnerCount, hotRatio,
                leafStats.size());
    }
}
