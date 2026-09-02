/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Abstract base class for all index nodes (both routing and data).
 * Holds the common structural properties: key range and slot capacity.
 */
public abstract class Node {
    protected final double lower;
    protected final double upper;
    protected final int capacity;

    protected Node(int capacity, double lower, double upper) {
        this.capacity = capacity;
        this.lower = lower;
        this.upper = upper;
    }

    public double lower() { return lower; }
    public double upper() { return upper; }
    public int capacity() { return capacity; }

    /**
     * Linear interpolation: maps a key to a slot index.
     * Subclasses may override for type-specific precision needs.
     */
    public int forward(double key) {
        double pos = capacity * (key - lower) / (upper - lower);
        return Math.min(capacity - 1, Math.max(0, (int) pos));
    }
}
