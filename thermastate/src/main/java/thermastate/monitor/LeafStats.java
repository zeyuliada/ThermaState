/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.monitor;

import thermastate.index.Leaf;

/**
 * Per-Leaf temperature statistics — raw observation only, no label.
 * readCount/writeCount are updated live during the window.
 * fr, fw, T_read, T_write are computed at window close.
 */
public class LeafStats {

    final Leaf node;
    final double lower;
    final double upper;
    long readCount;
    long writeCount;
    double fr;
    double fw;
    double T_read;
    double T_write;

    public LeafStats(Leaf node) {
        this(node, 0, 0);
    }

    /** Constructor with initial counts (for testing / synthetic data). */
    public LeafStats(Leaf node, long readCount, long writeCount) {
        this.node = node;
        this.lower = node.lower();
        this.upper = node.upper();
        this.readCount = readCount;
        this.writeCount = writeCount;
        this.fr = 0.0;
        this.fw = 0.0;
        this.T_read = 0.0;
        this.T_write = 0.0;
    }

    public Leaf node() { return node; }
    public double lower() { return lower; }
    public double upper() { return upper; }
    public long readCount() { return readCount; }
    public long writeCount() { return writeCount; }
    public long totalOps() { return readCount + writeCount; }
    public double fr() { return fr; }
    public double fw() { return fw; }
    public double T_read() { return T_read; }
    public double T_write() { return T_write; }

    public double range() { return upper - lower; }

    /** Number of keys stored in this leaf. */
    public int dataSize() { return node.size(); }

    @Override
    public String toString() {
        return String.format("LeafStats[range=[%.1f,%.1f] r=%d w=%d fr=%.4f fw=%.4f Tr=%.4f Tw=%.4f |D|=%d]",
                lower, upper, readCount, writeCount, fr, fw, T_read, T_write, dataSize());
    }
}
