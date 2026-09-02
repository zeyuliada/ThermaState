/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate;

/**
 * Paged flat storage for encoded-key → value bundles.
 *
 * Each bundle gets a unique int {@code bundleId} that encodes (pageIndex, offset).
 * New pages are appended on demand — existing data is never copied.
 *
 * <pre>
 * PAGE_SIZE = 1024, PAGE_SHIFT = 10, PAGE_MASK = 0x3FF
 * bundleId: [pageIndex (22 bits) | offset (10 bits)]
 * </pre>
 *
 * Deletion sets the value slot to null (logical delete); slots are never reclaimed.
 */
public class ValueBundleBlock<V> {

    public static final int PAGE_SHIFT = 10;
    public static final int PAGE_SIZE  = 1 << PAGE_SHIFT;
    public static final int PAGE_MASK  = PAGE_SIZE - 1;

    private static class Page<V> {
        final double[] keys;
        final V[] values;

        @SuppressWarnings("unchecked")
        Page() {
            this.keys   = new double[PAGE_SIZE];
            this.values = (V[]) new Object[PAGE_SIZE];
        }
    }

    private final java.util.ArrayList<Page<V>> pages;
    private int totalAllocated;

    public ValueBundleBlock() {
        this.pages = new java.util.ArrayList<>();
        this.totalAllocated = 0;
    }

    /** Allocate a new slot. Returns the bundle ID. */
    public int add(double encodedKey, V value) {
        int pageIdx = totalAllocated >>> PAGE_SHIFT;
        int offset  = totalAllocated & PAGE_MASK;
        if (offset == 0) {
            pages.add(new Page<>());
        }
        Page<V> p = pages.get(pageIdx);
        p.keys[offset]   = encodedKey;
        p.values[offset] = value;
        totalAllocated++;
        return (pageIdx << PAGE_SHIFT) | offset;
    }

    public V getValue(int bundleId) {
        int pageIdx = bundleId >>> PAGE_SHIFT;
        int offset  = bundleId & PAGE_MASK;
        if (pageIdx < 0 || pageIdx >= pages.size()) return null;
        return pages.get(pageIdx).values[offset];
    }

    public void setValue(int bundleId, V value) {
        int pageIdx = bundleId >>> PAGE_SHIFT;
        int offset  = bundleId & PAGE_MASK;
        if (pageIdx < pages.size()) {
            pages.get(pageIdx).values[offset] = value;
        }
    }

    public double getKey(int bundleId) {
        int pageIdx = bundleId >>> PAGE_SHIFT;
        int offset  = bundleId & PAGE_MASK;
        if (pageIdx < 0 || pageIdx >= pages.size()) {
            return Double.NaN;
        }
        return pages.get(pageIdx).keys[offset];
    }

    /** Total slots ever allocated (includes logically deleted). */
    public int totalAllocated() {
        return totalAllocated;
    }

    /** Count of non-null value slots (active bundles). */
    public int activeCount() {
        int n = 0;
        for (Page<V> p : pages) {
            for (int i = 0; i < PAGE_SIZE; i++) {
                if (p.values[i] != null) n++;
            }
        }
        return n;
    }
}
