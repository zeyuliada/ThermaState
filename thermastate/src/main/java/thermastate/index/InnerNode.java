/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Linear model routing node. Uses linear interpolation to map
 * a key to a child slot index. Each slot can point to either
 * another InnerNode (inner child) or a Leaf (data child).
 */
public class InnerNode extends Node {
    private final Node[] slots;
    private final int[] bitmap;
    private final int bitmapWords;

    InnerNode(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
        this.slots = new Node[capacity];
        this.bitmapWords = (capacity >> 5) + 1;
        this.bitmap = new int[bitmapWords];
    }

    public static InnerNode create(int capacity, double lower, double upper) {
        return new InnerNode(capacity, lower, upper);
    }

    public double[] subInterval(int position) {
        double slotInterval = (upper - lower) / (double) capacity;
        return new double[]{
            slotInterval * (double) position + lower,
            slotInterval * (double) (position + 1) + lower
        };
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

    /** Bit=1 → inner child; bit=0 → leaf (or empty). */
    public boolean isInnerSlot(int position) {
        return getBit(position);
    }

    public Node getInnerChild(int position) {
        return slots[position];
    }

    public Leaf getLeafChild(int position) {
        return (Leaf) slots[position];
    }

    public Node getChild(int position) {
        return slots[position];
    }

    public void setChild(int position, Node child) {
        if (child instanceof InnerNode || child instanceof ColdInnerNode) {
            setInnerChild(position, child);
        } else {
            setLeafChild(position, (Leaf) child);
        }
    }

    public void setInnerChild(int position, Node child) {
        slots[position] = child;
        setBit(position);
    }

    public void setLeafChild(int position, Leaf child) {
        slots[position] = child;
        clearBit(position);
    }

    public void forEachChild(ChildVisitor visitor) {
        for (int i = 0; i < capacity; i++) {
            if (getBit(i)) {
                visitor.visitInner(i, getInnerChild(i));
            } else if (slots[i] != null) {
                visitor.visitLeaf(i, getLeafChild(i));
            }
        }
    }

    public int childCount() {
        int count = 0;
        for (int i = 0; i < capacity; i++) {
            if (slots[i] != null) count++;
        }
        return count;
    }

    public interface ChildVisitor {
        void visitInner(int position, Node child);
        void visitLeaf(int position, Leaf child);
    }
}
