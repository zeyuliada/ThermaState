/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Standalone int-valued B+tree for ThermaState ColdLeaf.
 *
 * Internal nodes store routing keys; leaf nodes store (key, bundleId) pairs
 * linked for range scans. Root is always an InternalNode for uniform traversal.
 *
 * Parameters: ORDER = 64
 */
public class BPlusTree {

    public static final int ORDER = 64;
    static final int MIN_KEYS = ORDER / 2;

    static class Leaf {
        int numPairs;
        double[] keys = new double[ORDER];
        int[] values = new int[ORDER];
        Leaf next;

        int valAt(int i) { return values[i]; }
    }

    static class Internal {
        int numKeys;
        double[] keys = new double[ORDER];
        Object[] children = new Object[ORDER + 1];
        boolean leafChildren;
    }

    final double lower, upper;
    Internal root;
    int size;

    public BPlusTree(double lower, double upper) {
        this.lower = lower;
        this.upper = upper;
        this.root = null;
        this.size = 0;
    }

    public int size() { return size; }

    private Leaf searchLeaf(double key) {
        Internal cur = root;
        while (cur != null) {
            int pos = binarySearch(cur.keys, 0, cur.numKeys, key);
            if (cur.leafChildren) {
                return (Leaf) cur.children[pos];
            }
            cur = (Internal) cur.children[pos];
        }
        return null;
    }

    private static int binarySearch(double[] keys, int off, int len, double key) {
        int lo = off, hi = off + len - 1;
        int pos = off + len;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (key < keys[mid]) { hi = mid - 1; pos = mid; }
            else { lo = mid + 1; }
        }
        return pos;
    }

    private static int indexOf(double[] keys, int len, double key) {
        int lo = 0, hi = len - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >>> 1;
            if (keys[mid] == key) return mid;
            if (keys[mid] < key) lo = mid + 1;
            else hi = mid - 1;
        }
        return -1;
    }

    /** Returns bundleId or -1 if not found. */
    public int get(double key) {
        Leaf leaf = searchLeaf(key);
        if (leaf == null) return -1;
        int idx = indexOf(leaf.keys, leaf.numPairs, key);
        return idx >= 0 ? leaf.valAt(idx) : -1;
    }

    public void put(double key, int value) {
        if (root == null) {
            Leaf leaf = new Leaf();
            leaf.keys[0] = key;
            leaf.values[0] = value;
            leaf.numPairs = 1;
            root = new Internal();
            root.leafChildren = true;
            root.children[0] = leaf;
            root.numKeys = 0;
            size = 1;
            return;
        }

        Leaf leaf = searchLeaf(key);
        int idx = indexOf(leaf.keys, leaf.numPairs, key);
        if (idx >= 0) {
            leaf.values[idx] = value;
            return;
        }

        int pos = binarySearch(leaf.keys, 0, leaf.numPairs, key);
        System.arraycopy(leaf.keys, pos, leaf.keys, pos + 1, leaf.numPairs - pos);
        System.arraycopy(leaf.values, pos, leaf.values, pos + 1, leaf.numPairs - pos);
        leaf.keys[pos] = key;
        leaf.values[pos] = value;
        leaf.numPairs++;
        size++;

        if (leaf.numPairs >= ORDER) splitLeaf(leaf);
    }

    private void splitLeaf(Leaf left) {
        Leaf right = new Leaf();
        int mid = left.numPairs / 2;
        int moveCount = left.numPairs - mid;
        System.arraycopy(left.keys, mid, right.keys, 0, moveCount);
        System.arraycopy(left.values, mid, right.values, 0, moveCount);
        right.numPairs = moveCount;
        left.numPairs = mid;
        right.next = left.next;
        left.next = right;

        double sep = right.keys[0];
        insertInParent(left, sep, right);
    }

    private void insertInParent(Leaf child, double sep, Leaf newChild) {
        Internal parent = findParentOf(child);
        if (parent == null) {
            Internal newParent = insertIntoNode(root, child, sep, newChild);
            if (newParent != null) {
                Internal newRoot = new Internal();
                newRoot.leafChildren = false;
                newRoot.keys[0] = subtreeMin(newParent);
                newRoot.children[0] = root;
                newRoot.children[1] = newParent;
                newRoot.numKeys = 1;
                root = newRoot;
            }
        } else {
            Internal split = insertIntoNode(parent, child, sep, newChild);
            while (split != null) {
                Internal grandparent = findParentOf(parent);
                if (grandparent == null) {
                    Internal newRoot = new Internal();
                    newRoot.leafChildren = false;
                    newRoot.keys[0] = subtreeMin(split);
                    newRoot.children[0] = parent;
                    newRoot.children[1] = split;
                    newRoot.numKeys = 1;
                    root = newRoot;
                    break;
                }
                double upSep = subtreeMin(split);
                split = insertIntoNode(grandparent, parent, upSep, split);
                parent = grandparent;
            }
        }
    }

    private Internal findParentOf(Object child) {
        if (root == null || root.leafChildren) return null;
        return findParentRecursive(root, child);
    }

    private Internal findParentRecursive(Internal node, Object child) {
        for (int i = 0; i <= node.numKeys; i++) {
            if (node.children[i] == child) return node;
        }
        if (!node.leafChildren) {
            for (int i = 0; i <= node.numKeys; i++) {
                Internal found = findParentRecursive((Internal) node.children[i], child);
                if (found != null) return found;
            }
        }
        return null;
    }

    private Internal insertIntoNode(Internal node, Object child, double sep, Object newChild) {
        int pos = -1;
        for (int i = 0; i <= node.numKeys; i++) {
            if (node.children[i] == child) { pos = i; break; }
        }
        if (pos < 0) return null;

        System.arraycopy(node.keys, pos, node.keys, pos + 1, node.numKeys - pos);
        node.keys[pos] = sep;
        System.arraycopy(node.children, pos + 1, node.children, pos + 2, node.numKeys - pos);
        node.children[pos + 1] = newChild;
        node.numKeys++;

        if (node.numKeys >= ORDER) return splitInternal(node);
        return null;
    }

    private Internal splitInternal(Internal left) {
        Internal right = new Internal();
        right.leafChildren = left.leafChildren;
        int mid = left.numKeys / 2;
        int moveKeys = left.numKeys - mid - 1;
        int moveChildren = left.numKeys - mid;
        System.arraycopy(left.keys, mid + 1, right.keys, 0, moveKeys);
        System.arraycopy(left.children, mid + 1, right.children, 0, moveChildren);
        right.numKeys = moveKeys;
        left.numKeys = mid;
        return right;
    }

    private static double subtreeMin(Internal node) {
        while (!node.leafChildren) {
            node = (Internal) node.children[0];
        }
        Leaf leaf = (Leaf) node.children[0];
        return leaf.numPairs > 0 ? leaf.keys[0] : 0;
    }

    private static double subtreeMax(Internal node) {
        while (!node.leafChildren) {
            node = (Internal) node.children[node.numKeys];
        }
        Leaf leaf = (Leaf) node.children[node.numKeys];
        return leaf.numPairs > 0 ? leaf.keys[leaf.numPairs - 1] : 0;
    }

    /** Bulk-load from sorted parallel arrays. */
    public void bulkLoad(double[] keys, int[] values) {
        if (keys.length == 0) return;
        size = keys.length;

        java.util.ArrayList<Leaf> leaves = new java.util.ArrayList<>();
        int i = 0;
        while (i < keys.length) {
            Leaf leaf = new Leaf();
            int cnt = 0;
            while (cnt < ORDER && i < keys.length) {
                leaf.keys[cnt] = keys[i];
                leaf.values[cnt] = values[i];
                cnt++; i++;
            }
            leaf.numPairs = cnt;
            leaves.add(leaf);
        }
        for (int j = 0; j + 1 < leaves.size(); j++) {
            leaves.get(j).next = leaves.get(j + 1);
        }

        java.util.ArrayList<Object> level = new java.util.ArrayList<>(leaves);
        boolean leafLevel = true;

        while (level.size() > 1) {
            java.util.ArrayList<Object> nextLevel = new java.util.ArrayList<>();
            int j = 0;
            while (j < level.size()) {
                Internal in = new Internal();
                in.leafChildren = leafLevel;
                int cnt = 0;
                double maxK = 0;
                while (cnt < ORDER + 1 && j < level.size()) {
                    double childMax = leafLevel
                        ? leafMaxKey((Leaf) level.get(j))
                        : subtreeMax((Internal) level.get(j));
                    if (cnt > 0) in.keys[cnt - 1] = maxK;
                    in.children[cnt] = level.get(j);
                    maxK = childMax;
                    cnt++; j++;
                }
                in.numKeys = cnt - 1;
                nextLevel.add(in);
            }
            level = nextLevel;
            leafLevel = false;
        }

        root = (Internal) level.get(0);
    }

    private static double leafMaxKey(Leaf leaf) {
        return leaf.numPairs > 0 ? leaf.keys[leaf.numPairs - 1] : 0;
    }

    public boolean remove(double key) {
        Leaf leaf = searchLeaf(key);
        if (leaf == null) return false;
        int idx = indexOf(leaf.keys, leaf.numPairs, key);
        if (idx < 0) return false;
        int after = leaf.numPairs - idx - 1;
        if (after > 0) {
            System.arraycopy(leaf.keys, idx + 1, leaf.keys, idx, after);
            System.arraycopy(leaf.values, idx + 1, leaf.values, idx, after);
        }
        leaf.values[leaf.numPairs - 1] = 0;
        leaf.numPairs--;
        size--;
        return true;
    }

    public Leaf firstLeaf() {
        if (root == null) return null;
        Internal cur = root;
        while (!cur.leafChildren) cur = (Internal) cur.children[0];
        return (Leaf) cur.children[0];
    }

    public long memoryUsed() {
        return countBytes(root);
    }

    private static long countBytes(Internal node) {
        if (node == null) return 0;
        long bytes = 40L + (long) ORDER * 8L + (long) (ORDER + 1) * 4L;
        if (node.leafChildren) {
            for (int i = 0; i <= node.numKeys; i++) {
                if (node.children[i] != null) {
                    bytes += 40L + (long) ORDER * 8L + (long) ORDER * 4L;
                }
            }
        } else {
            for (int i = 0; i <= node.numKeys; i++) {
                bytes += countBytes((Internal) node.children[i]);
            }
        }
        return bytes;
    }
}
