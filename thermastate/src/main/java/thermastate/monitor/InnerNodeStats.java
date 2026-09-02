/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.monitor;

import thermastate.index.InnerNode;

/**
 * Per-InnerNode temperature statistics.
 * routeCount is updated live during the window. Temperature scores
 * and labels are computed at window close (only for level &lt; h).
 */
public class InnerNodeStats {

    public enum Label { HOT, COLD }

    final InnerNode node;
    final int level;
    final double lower;
    final double upper;
    long routeCount;
    double T_route;
    Label label;

    InnerNodeStats(InnerNode node, int level) {
        this.node = node;
        this.level = level;
        this.lower = node.lower();
        this.upper = node.upper();
        this.routeCount = 0;
        this.T_route = 0.0;
        this.label = null;
    }

    public InnerNode node() { return node; }
    public int level() { return level; }
    public double lower() { return lower; }
    public double upper() { return upper; }
    public long routeCount() { return routeCount; }
    public double T_route() { return T_route; }
    public Label label() { return label; }
    public boolean isHot() { return label == Label.HOT; }
    public boolean isCold() { return label == Label.COLD; }

    public double range() { return upper - lower; }

    @Override
    public String toString() {
        return String.format("InnerStats[lv=%d range=[%.1f,%.1f] routes=%d T=%.4f %s]",
                level, lower, upper, routeCount, T_route, label);
    }
}
