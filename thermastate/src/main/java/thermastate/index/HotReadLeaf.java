/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Hot-read-optimized leaf node.
 *
 * Maintains a contiguous sorted key array [0, size-1] with a learned linear model.
 * Lookup: model predicts rank → local scan (≤ 8 slots).
 * Insert: model predicts rank → shift right (≤ 16 elements) → insert.
 */
public class HotReadLeaf extends Leaf {

    private static final int SCAN_RANGE = 8;
    private static final int SHIFT_RANGE = 16;

    HotReadLeaf(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
    }

    public static HotReadLeaf create(int capacity, double lower, double upper) {
        return new HotReadLeaf(capacity, lower, upper);
    }

    @Override
    public LeafType leafType() { return LeafType.HOT_READ; }

    private int predictRank(double key) {
        if (size == 0) return 0;
        double kMin = keys[0];
        double kMax = keys[size - 1];
        if (kMax <= kMin) return 0;
        double pos = size * (key - kMin) / (kMax - kMin);
        int p = (int) pos;
        if (p < 0) p = 0;
        if (p >= size) p = size - 1;
        return p;
    }

    @Override
    public int get(double key) {
        int pos = predictRank(key);
        for (int i = pos, step = 0; step <= SCAN_RANGE && i >= 0; step++, i--) {
            if (keys[i] == key) return values[i];
        }
        for (int i = pos + 1, step = 1; step <= SCAN_RANGE && i < size; step++, i++) {
            if (keys[i] == key) return values[i];
        }
        return -1;
    }

    @Override
    public boolean put(double key, int value) {
        if (size == 0) {
            keys[0] = key; values[0] = value; size = 1; return true;
        }

        int pred = predictRank(key);
        int insertPos = -1;
        int scanL = Math.max(0, pred - SCAN_RANGE);
        int scanR = Math.min(size, pred + SCAN_RANGE);

        for (int i = scanL; i < scanR; i++) {
            if (keys[i] == key) { values[i] = value; return false; }
            if (keys[i] > key) { insertPos = i; break; }
        }
        if (insertPos < 0) insertPos = size;

        if (size >= capacity) return false;

        int shiftCount = size - insertPos;
        if (shiftCount > SHIFT_RANGE) return false;

        if (shiftCount > 0) {
            System.arraycopy(keys, insertPos, keys, insertPos + 1, shiftCount);
            System.arraycopy(values, insertPos, values, insertPos + 1, shiftCount);
        }
        keys[insertPos] = key;
        values[insertPos] = value;
        size++;
        return true;
    }

    @Override
    public boolean erase(double key) {
        int pos = predictRank(key);
        int found = -1;
        for (int i = pos, step = 0; step <= SCAN_RANGE && i >= 0; step++, i--) {
            if (keys[i] == key) { found = i; break; }
        }
        if (found < 0) {
            for (int i = pos + 1, step = 1; step <= SCAN_RANGE && i < size; step++, i++) {
                if (keys[i] == key) { found = i; break; }
            }
        }
        if (found < 0) return false;

        int after = size - found - 1;
        if (after > 0) {
            System.arraycopy(keys, found + 1, keys, found, after);
            System.arraycopy(values, found + 1, values, found, after);
        }
        values[size - 1] = 0;
        size--;
        return true;
    }

    public void bulkInsert(double key, int value) {
        if (size >= capacity) {
            throw new RuntimeException("HotReadLeaf bulkInsert: capacity exceeded");
        }
        keys[size] = key;
        values[size] = value;
        size++;
    }

    @Override
    public Leaf expand() {
        int newCap = Math.max((int) (size / DataNode.DEFAULT_DENSITY), capacity * 2);
        HotReadLeaf newLeaf = new HotReadLeaf(newCap, lower, upper);
        for (int i = 0; i < size; i++) {
            newLeaf.keys[i] = keys[i];
            newLeaf.values[i] = values[i];
        }
        newLeaf.size = size;
        return newLeaf;
    }

    @Override
    public long memoryUsed() {
        return (long) capacity * 8L + (long) capacity * 4L + 32L;
    }

    @Override
    public void forEachEntry(IntEntryVisitor visitor) {
        for (int i = 0; i < size; i++) {
            visitor.visit(keys[i], values[i]);
        }
    }
}
