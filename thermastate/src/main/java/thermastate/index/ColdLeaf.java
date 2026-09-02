/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * B+tree-backed cold leaf node — space-efficient, no learned model.
 * Delegates all key-value storage to an internal int-valued BPlusTree.
 */
public class ColdLeaf extends Leaf {

    private ColdLeaf nextLeaf;
    private final BPlusTree tree;

    ColdLeaf(int capacity, double lower, double upper) {
        super(Math.max(1, capacity), lower, upper);
        this.nextLeaf = null;
        this.tree = new BPlusTree(lower, upper);
    }

    public static ColdLeaf create(int capacity, double lower, double upper) {
        return new ColdLeaf(capacity, lower, upper);
    }

    public ColdLeaf nextLeaf() { return nextLeaf; }

    @Override
    public LeafType leafType() { return LeafType.COLD; }

    @Override
    public int get(double key) { return tree.get(key); }

    @Override
    public boolean put(double key, int value) {
        tree.put(key, value);
        return true;
    }

    @Override
    public boolean erase(double key) { return tree.remove(key); }

    @Override
    public int size() { return tree.size(); }

    @Override
    public long memoryUsed() { return tree.memoryUsed(); }

    @Override
    public boolean isFull() { return false; }

    public void bulkInsert(double key, int value) { tree.put(key, value); }

    /** Split by moving entries above midpoint to a new ColdLeaf. */
    public ColdLeaf split() {
        int total = tree.size();
        int mid = total / 2;
        int count = 0;
        double splitKey = lower;

        BPlusTree.Leaf leaf = tree.firstLeaf();
        while (leaf != null && count + leaf.numPairs <= mid) {
            count += leaf.numPairs;
            leaf = leaf.next;
        }
        if (leaf != null && leaf.numPairs > 0) {
            splitKey = leaf.keys[0];
        }

        ColdLeaf newLeaf = create(capacity, splitKey, upper);

        BPlusTree.Leaf srcLeaf = tree.firstLeaf();
        while (srcLeaf != null) {
            for (int i = 0; i < srcLeaf.numPairs; i++) {
                if (srcLeaf.keys[i] >= splitKey && count <= total) {
                    newLeaf.tree.put(srcLeaf.keys[i], srcLeaf.valAt(i));
                    tree.remove(srcLeaf.keys[i]);
                }
            }
            srcLeaf = srcLeaf.next;
        }

        newLeaf.nextLeaf = this.nextLeaf;
        this.nextLeaf = newLeaf;
        return newLeaf;
    }

    @Override
    public Leaf expand() {
        int newCap = Math.max((int) (tree.size() / DataNode.DEFAULT_DENSITY), capacity * 2);
        ColdLeaf newLeaf = new ColdLeaf(newCap, lower, upper);
        BPlusTree.Leaf leaf = tree.firstLeaf();
        while (leaf != null) {
            for (int i = 0; i < leaf.numPairs; i++) {
                newLeaf.tree.put(leaf.keys[i], leaf.valAt(i));
            }
            leaf = leaf.next;
        }
        newLeaf.nextLeaf = nextLeaf;
        return newLeaf;
    }

    @Override
    public void forEachEntry(IntEntryVisitor visitor) {
        BPlusTree.Leaf leaf = tree.firstLeaf();
        while (leaf != null) {
            for (int i = 0; i < leaf.numPairs; i++) {
                visitor.visit(leaf.keys[i], leaf.valAt(i));
            }
            leaf = leaf.next;
        }
    }
}
