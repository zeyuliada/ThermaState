/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Open-addressing hash table leaf node. Uses a learned linear model
 * to predict the hash position, then probes bidirectionally.
 */
public class DataNode extends Leaf {
    private static final double HASH_FACTOR = 10231023.353;
    private static final double MIN_DENSITY = 0.35;
    static final double DEFAULT_DENSITY = 0.65;
    private static final double MAX_DENSITY = 0.95;

    public static final int DATA_NODE_SIZE = 5;

    private final int[] bitmap;
    private final int bitmapWords;
    private int maxOffset;

    public DataNode(int capacity, double lower, double upper) {
        super(capacity, lower, upper);
        this.bitmapWords = (capacity >> 5) + 1;
        this.bitmap = new int[bitmapWords];
        this.maxOffset = 0;
    }

    public static DataNode create(int capacity, double lower, double upper) {
        return new DataNode(capacity, lower, upper);
    }

    private double hashForward(double key) {
        double pos = capacity * ((key - lower) / (upper - lower));
        return Math.min(capacity - 1, Math.max(0.0, pos));
    }

    private boolean getBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        return (bitmap[index] >> bitIndex & 1) == 1;
    }

    private void setBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        bitmap[index] |= 1 << bitIndex;
    }

    private void clearBit(int position) {
        int index = position >> 5;
        int bitIndex = position - (index << 5);
        bitmap[index] &= ~(1 << bitIndex);
    }

    private int hash(double key) {
        long predictHash = (long) (hashForward(key) * HASH_FACTOR);
        return (int) (predictHash % (long) capacity);
    }

    public int find(double key) {
        int position = hash(key);
        int left = position;
        int right = position;
        for (int i = 0; i <= maxOffset; i++, left--, right++) {
            if (left < 0) left += capacity;
            if (right >= capacity) right -= capacity;
            if (getBit(left) && keys[left] == key) return left;
            if (getBit(right) && keys[right] == key) return right;
        }
        return -1;
    }

    public int findInsert(double key) {
        int position = hash(key);
        int halfLen = (capacity >> 1) + 1;
        int left = position;
        int right = position;

        int i = 0;
        for (; i <= maxOffset; i++, left--, right++) {
            if (left < 0) left += capacity;
            if (right >= capacity) right -= capacity;
            if (!getBit(left)) return left;
            if (keys[left] == key) return -1;
            if (!getBit(right)) return right;
            if (keys[right] == key) return -1;
        }

        for (; i <= halfLen; i++, left--, right++) {
            if (left < 0) left += capacity;
            if (right >= capacity) right -= capacity;
            if (!getBit(left)) { maxOffset = i; return left; }
            if (!getBit(right)) { maxOffset = i; return right; }
        }
        return -1; // full
    }

    public double getKey(int position) { return keys[position]; }
    public int getValue(int position) { return values[position]; }

    public void insertAt(int position, double key, int value) {
        keys[position] = key;
        values[position] = value;
        setBit(position);
        size++;
    }

    public void removeAt(int position) {
        clearBit(position);
        keys[position] = 0;
        values[position] = 0;
        size--;
    }

    public boolean isUnderMin() {
        return capacity > DATA_NODE_SIZE && size < capacity * MIN_DENSITY;
    }

    @Override
    public Leaf expand() {
        return expand(this);
    }

    static DataNode expand(DataNode old) {
        int newCapacity = (int) Math.max(old.size / DEFAULT_DENSITY, DATA_NODE_SIZE);
        DataNode newNode = create(newCapacity, old.lower, old.upper);
        for (int i = 0; i < old.capacity; i++) {
            if (old.getBit(i)) {
                int pos = newNode.findInsert(old.keys[i]);
                newNode.insertAt(pos, old.keys[i], old.values[i]);
            }
        }
        return newNode;
    }

    static DataNode shrink(DataNode old) {
        int newCapacity = Math.max((int) (old.size / DEFAULT_DENSITY), DATA_NODE_SIZE);
        DataNode newNode = create(newCapacity, old.lower, old.upper);
        for (int i = 0; i < old.capacity; i++) {
            if (old.getBit(i)) {
                int pos = newNode.findInsert(old.keys[i]);
                newNode.insertAt(pos, old.keys[i], old.values[i]);
            }
        }
        return newNode;
    }

    @Override
    public int get(double key) {
        int pos = find(key);
        return pos < 0 ? -1 : values[pos];
    }

    @Override
    public boolean put(double key, int value) {
        int pos = findInsert(key);
        if (pos < 0) return false;
        insertAt(pos, key, value);
        return true;
    }

    @Override
    public boolean erase(double key) {
        int pos = find(key);
        if (pos < 0) return false;
        removeAt(pos);
        return true;
    }

    @Override
    public long memoryUsed() {
        return (long) capacity * 8L
             + (long) capacity * 4L
             + (long) bitmapWords * 4L
             + 24L;
    }

    @Override
    public LeafType leafType() {
        return LeafType.COLD;
    }

    @Override
    public void forEachEntry(IntEntryVisitor visitor) {
        for (int i = 0; i < capacity; i++) {
            if (getBit(i)) {
                visitor.visit(keys[i], values[i]);
            }
        }
    }
}
