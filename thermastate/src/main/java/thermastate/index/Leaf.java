/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * Abstract base for all data-storing leaf nodes.
 *
 * Stores {@code double[] keys} and {@code int[] values} (bundle IDs).
 * Subclasses: DataNode, HotReadLeaf, HotWriteLeaf, ColdLeaf.
 */
public abstract class Leaf extends Node {

    protected final double[] keys;
    protected final int[] values;
    protected int size;

    /**
     * Non-null while a delta-log rebuild is in progress on a HotWriteLeaf.
     * Concurrent writes are redirected here; reads check it as an override layer.
     * Set/cleared by {@code Reorganizer.rebuildHotWriteWithDelta}.
     * ColdLeaf / HotReadLeaf / DataNode never touch this.
     */
    public volatile DeltaBuffer pendingDelta;

    protected Leaf(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
        this.keys = new double[capacity];
        this.values = new int[capacity];
        this.size = 0;
    }

    public int size() { return size; }

    // ── Subclass contract ──

    /** Returns bundleId or -1 if not found. */
    public abstract int get(double key);

    /** Insert or update. Returns false if key already exists or leaf is full. */
    public abstract boolean put(double key, int value);

    public abstract boolean erase(double key);

    public abstract long memoryUsed();

    public abstract LeafType leafType();

    /** Create a new leaf of the same type with larger capacity, copying all entries. */
    public abstract Leaf expand();

    public boolean isFull() {
        return size > capacity * 0.95;
    }

    /** Total entries stored, including overflow chains (HotWriteLeaf overrides). */
    public int totalEntries() { return size; }

    /** Sort entries [0, size) by key in-place. Used after bulk-inserting delta entries. */
    void sortByKey() {
        if (size <= 1) return;
        // Create index permutation array, sort by key, then permute
        Integer[] idx = new Integer[size];
        for (int i = 0; i < size; i++) idx[i] = i;
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(keys[a], keys[b]));

        double[] sortedKeys = new double[size];
        int[] sortedVals = new int[size];
        for (int i = 0; i < size; i++) {
            sortedKeys[i] = keys[idx[i]];
            sortedVals[i] = values[idx[i]];
        }
        System.arraycopy(sortedKeys, 0, keys, 0, size);
        System.arraycopy(sortedVals, 0, values, 0, size);
    }

    public abstract void forEachEntry(IntEntryVisitor visitor);

    public interface IntEntryVisitor {
        void visit(double key, int bundleId);
    }

    // ── Delta log (HotWriteLeaf rebuild) ─────────────────────────────

    /**
     * Append-only concurrent write buffer.  Used during HotWriteLeaf rebuild
     * so that writes are not blocked while the new leaf is being constructed.
     *
     * Lifecycle: created → set on leaf → writes redirect here → snapshot →
     * new leaf built → drain & replay → cleared from leaf.
     */
    public static final class DeltaBuffer {

        private final ConcurrentLinkedQueue<WriteOp> ops = new ConcurrentLinkedQueue<>();

        static final class WriteOp {
            final double key;
            final int bundleId;       // ignored when isErase=true
            final boolean isErase;

            WriteOp(double key, int bundleId, boolean isErase) {
                this.key = key;
                this.bundleId = bundleId;
                this.isErase = isErase;
            }
        }

        public void addPut(double key, int bundleId) {
            ops.add(new WriteOp(key, bundleId, false));
        }

        public void addErase(double key) {
            ops.add(new WriteOp(key, 0, true));
        }

        public boolean isEmpty() { return ops.isEmpty(); }

        public int opCount() { return ops.size(); }

        /**
         * Returns the latest state for {@code key} in the delta:
         * {@code null} = not present, {@code -1} = erased, {@code ≥0} = bundleId.
         */
        public Integer find(double key) {
            Integer result = null;
            for (WriteOp op : ops) {
                if (op.key == key) {
                    result = op.isErase ? -1 : op.bundleId;
                }
            }
            return result;
        }

        /**
         * Drain all ops and replay onto {@code target}.
         *
         * For non-HotWriteLeaf targets, entries are re-sorted after replay
         * because {@code Index.bulkInsertInto} appends and we must preserve
         * the leaf's sorted-key invariant. HotWriteLeaf handles out-of-order
         * insertion natively via its overflow chain.
         */
        public void replay(Leaf target) {
            if (target.leafType() == LeafType.HOT_WRITE) {
                WriteOp op;
                while ((op = ops.poll()) != null) {
                    if (op.isErase) target.erase(op.key);
                    else target.put(op.key, op.bundleId);
                }
            } else {
                // Bulk-insert with sorting — works for HotReadLeaf / ColdLeaf
                java.util.ArrayList<double[]> puts = new java.util.ArrayList<>();
                java.util.ArrayList<WriteOp> erases = new java.util.ArrayList<>();
                WriteOp op;
                while ((op = ops.poll()) != null) {
                    if (op.isErase) erases.add(op);
                    else puts.add(new double[]{op.key, (double) op.bundleId});
                }
                puts.sort((a, b) -> Double.compare(a[0], b[0]));
                for (double[] p : puts) {
                    Index.bulkInsertInto(target, p[0], (int) p[1]);
                }
                // Re-sort the full leaf so delta entries are in correct position
                target.sortByKey();
                for (WriteOp e : erases) {
                    target.erase(e.key);
                }
            }
        }
    }
}
