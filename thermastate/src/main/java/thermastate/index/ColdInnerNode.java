/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * B+tree-style internal node. Uses separator keys + binary search to route
 * keys, instead of a learned linear model.
 */
public class ColdInnerNode extends Node {
    private final Node[] slots;
    private final double[] separators;
    private final int[] bitmap;
    private final int bitmapWords;

    ColdInnerNode(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
        this.slots = new Node[capacity];
        this.separators = new double[capacity - 1];
        this.bitmapWords = (capacity >> 5) + 1;
        this.bitmap = new int[bitmapWords];
    }

    public static ColdInnerNode create(int capacity, double lower, double upper) {
        return new ColdInnerNode(capacity, lower, upper);
    }

    @Override
    public int forward(double key) {
        int lo = 0, hi = capacity - 1;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (key < separators[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    }

    public double[] subInterval(int position) {
        double lo = (position == 0) ? lower : separators[position - 1];
        double hi = (position == capacity - 1) ? upper : separators[position];
        return new double[]{lo, hi};
    }

    private boolean getBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        return (bitmap[index] >> bitIndex & 1) == 1;
    }

    void setBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        bitmap[index] |= 1 << bitIndex;
    }

    void clearBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        bitmap[index] &= ~(1 << bitIndex);
    }

    public boolean isInnerSlot(int position) {
        return getBit(position);
    }

    public Node getInnerChild(int position) {
        return slots[position];
    }

    public ColdInnerNode getColdInnerChild(int position) {
        return (ColdInnerNode) slots[position];
    }

    public Leaf getLeafChild(int position) {
        return (Leaf) slots[position];
    }

    public Node getChild(int position) {
        return slots[position];
    }

    public void setChild(int position, Node child) {
        slots[position] = child;
        if (child instanceof InnerNode || child instanceof ColdInnerNode) {
            setBit(position);
        } else {
            clearBit(position);
        }
        if (position < capacity - 1) {
            separators[position] = child.upper();
        }
    }

    public int childCount() {
        int count = 0;
        for (int i = 0; i < capacity; i++) {
            if (slots[i] != null) count++;
        }
        return count;
    }

    public void forEachChild(ChildVisitor visitor) {
        for (int i = 0; i < capacity; i++) {
            if (slots[i] == null) continue;
            visitor.visit(i, slots[i]);
        }
    }

    public interface ChildVisitor {
        void visit(int position, Node child);
    }
}
