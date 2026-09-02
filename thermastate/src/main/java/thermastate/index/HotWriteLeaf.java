/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Hot-write-optimized leaf node.
 *
 * Sorted key array with overflow chains for write-heavy regions.
 * Model predicts rank; insert conflicts are chained rather than shifted.
 */
public class HotWriteLeaf extends Leaf {

    private static final int SCAN_RANGE = 8;
    private static final int MAX_SHIFT = 4;

    private final Overflow[] overflows;

    private static class Overflow {
        double key;
        int value;
        Overflow next;
        Overflow(double key, int value, Overflow next) {
            this.key = key; this.value = value; this.next = next;
        }
    }

    HotWriteLeaf(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
        this.overflows = new Overflow[capacity];
    }

    public static HotWriteLeaf create(int capacity, double lower, double upper) {
        return new HotWriteLeaf(capacity, lower, upper);
    }

    @Override
    public LeafType leafType() { return LeafType.HOT_WRITE; }

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
        int result = getInternal(key);
        DeltaBuffer delta = pendingDelta;
        if (delta != null) {
            Integer override = delta.find(key);
            if (override != null) return override;  // -1=erased, ≥0=bundleId
        }
        return result;
    }

    private int getInternal(double key) {
        int pos = predictRank(key);
        for (int i = pos, step = 0; step <= SCAN_RANGE && i >= 0; step++, i--) {
            if (keys[i] == key) return values[i];
            int found = searchOverflow(i, key);
            if (found != -1) return found;
        }
        for (int i = pos + 1, step = 1; step <= SCAN_RANGE && i < size; step++, i++) {
            if (keys[i] == key) return values[i];
            int found = searchOverflow(i, key);
            if (found != -1) return found;
        }
        return -1;
    }

    private int searchOverflow(int slot, double key) {
        for (Overflow ov = overflows[slot]; ov != null; ov = ov.next) {
            if (ov.key == key) return ov.value;
        }
        return -1;
    }

    @Override
    public boolean put(double key, int value) {
        DeltaBuffer delta = pendingDelta;
        if (delta != null) {
            delta.addPut(key, value);
            return true;
        }
        return putInternal(key, value);
    }

    private boolean putInternal(double key, int value) {
        if (size == 0) {
            keys[0] = key; values[0] = value; size = 1; return true;
        }

        int pred = predictRank(key);
        int scanL = Math.max(0, pred - SCAN_RANGE);
        int scanR = Math.min(size, pred + SCAN_RANGE);

        for (int i = scanL; i < scanR; i++) {
            if (keys[i] == key) { values[i] = value; return false; }
            for (Overflow ov = overflows[i]; ov != null; ov = ov.next) {
                if (ov.key == key) { ov.value = value; return false; }
            }
        }

        int insertPos = size;
        for (int i = scanL; i < scanR; i++) {
            if (keys[i] > key) { insertPos = i; break; }
        }

        int shiftCount = insertPos < size ? size - insertPos : 0;

        if (shiftCount <= MAX_SHIFT && size < capacity) {
            if (shiftCount > 0) {
                System.arraycopy(keys, insertPos, keys, insertPos + 1, shiftCount);
                System.arraycopy(values, insertPos, values, insertPos + 1, shiftCount);
                System.arraycopy(overflows, insertPos, overflows, insertPos + 1, shiftCount);
                overflows[insertPos] = null;
            }
            keys[insertPos] = key;
            values[insertPos] = value;
            size++;
            return true;
        }

        int chainTo = findChainSlot(insertPos);
        if (chainTo < 0) return false;
        overflows[chainTo] = new Overflow(key, value, overflows[chainTo]);
        return true;
    }

    private int findChainSlot(int nearPos) {
        if (nearPos < size && values[nearPos] != 0) return nearPos;
        for (int d = 1; d <= SCAN_RANGE; d++) {
            int l = nearPos - d;
            if (l >= 0 && values[l] != 0) return l;
            int r = nearPos + d;
            if (r < size && values[r] != 0) return r;
        }
        return -1;
    }

    @Override
    public boolean erase(double key) {
        DeltaBuffer delta = pendingDelta;
        if (delta != null) {
            delta.addErase(key);
            return true;
        }
        return eraseInternal(key);
    }

    private boolean eraseInternal(double key) {
        int pos = predictRank(key);
        for (int i = pos, step = 0; step <= SCAN_RANGE && i >= 0; step++, i--) {
            if (keys[i] == key) { shiftErase(i); return true; }
            if (removeFromOverflow(i, key)) return true;
        }
        for (int i = pos + 1, step = 1; step <= SCAN_RANGE && i < size; step++, i++) {
            if (keys[i] == key) { shiftErase(i); return true; }
            if (removeFromOverflow(i, key)) return true;
        }
        return false;
    }

    private void shiftErase(int idx) {
        int after = size - idx - 1;
        if (after > 0) {
            System.arraycopy(keys, idx + 1, keys, idx, after);
            System.arraycopy(values, idx + 1, values, idx, after);
            System.arraycopy(overflows, idx + 1, overflows, idx, after);
        }
        values[size - 1] = 0;
        overflows[size - 1] = null;
        size--;
    }

    private boolean removeFromOverflow(int slot, double key) {
        Overflow prev = null;
        for (Overflow ov = overflows[slot]; ov != null; prev = ov, ov = ov.next) {
            if (ov.key == key) {
                if (prev == null) overflows[slot] = ov.next;
                else prev.next = ov.next;
                return true;
            }
        }
        return false;
    }

    public void bulkInsert(double key, int value) {
        if (size >= capacity) {
            throw new RuntimeException("HotWriteLeaf bulkInsert: capacity exceeded");
        }
        keys[size] = key;
        values[size] = value;
        size++;
    }

    @Override
    public Leaf expand() {
        int newCap = Math.max((int) (size / DataNode.DEFAULT_DENSITY), capacity * 2);
        HotWriteLeaf newLeaf = new HotWriteLeaf(newCap, lower, upper);
        for (int i = 0; i < size; i++) {
            newLeaf.keys[i] = keys[i];
            newLeaf.values[i] = values[i];
            newLeaf.overflows[i] = overflows[i];
        }
        newLeaf.size = size;
        return newLeaf;
    }

    @Override
    public int totalEntries() {
        int total = size;
        for (int i = 0; i < size; i++) {
            for (Overflow ov = overflows[i]; ov != null; ov = ov.next) total++;
        }
        return total;
    }

    @Override
    public long memoryUsed() {
        long overflowBytes = 0;
        for (int i = 0; i < size; i++) {
            for (Overflow ov = overflows[i]; ov != null; ov = ov.next) {
                overflowBytes += 24;
            }
        }
        return (long) capacity * 8L + (long) capacity * 4L + 32L + overflowBytes;
    }

    @Override
    public void forEachEntry(IntEntryVisitor visitor) {
        for (int i = 0; i < size; i++) {
            visitor.visit(keys[i], values[i]);
            for (Overflow ov = overflows[i]; ov != null; ov = ov.next) {
                visitor.visit(ov.key, ov.value);
            }
        }
    }
}
