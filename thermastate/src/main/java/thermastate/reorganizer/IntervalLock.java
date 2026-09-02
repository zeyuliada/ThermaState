/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import thermastate.index.Node;

/**
 * Fine-grained interval-level read-write lock.
 *
 * Each registered interval root gets its own lock.  Read operations
 * (get/put) acquire the read lock; rebuild + swap acquires the write lock.
 * Non-overlapping intervals do not block each other.
 *
 * Paper Definition 4: Interval Lock.
 */
public final class IntervalLock {

    private final ConcurrentHashMap<Node, ReentrantReadWriteLock> locks;

    public IntervalLock() {
        this.locks = new ConcurrentHashMap<>();
    }

    /** Register a new interval root (called once after tree build). */
    public void register(Node intervalRoot) {
        locks.putIfAbsent(intervalRoot, new ReentrantReadWriteLock());
    }

    /** Register all nodes from a collection. */
    public void registerAll(Iterable<? extends Node> nodes) {
        for (Node n : nodes) register(n);
    }

    // ── Read lock (normal get/put) ──

    public void readLock(Node intervalRoot) {
        ReentrantReadWriteLock l = locks.get(intervalRoot);
        if (l != null) l.readLock().lock();
    }

    public void readUnlock(Node intervalRoot) {
        ReentrantReadWriteLock l = locks.get(intervalRoot);
        if (l != null) l.readLock().unlock();
    }

    // ── Write lock (rebuild + swap) ──

    public void writeLock(Node intervalRoot) {
        ReentrantReadWriteLock l = locks.get(intervalRoot);
        if (l != null) l.writeLock().lock();
    }

    public void writeUnlock(Node intervalRoot) {
        ReentrantReadWriteLock l = locks.get(intervalRoot);
        if (l != null) l.writeLock().unlock();
    }

    // ── Stats ──

    public int lockCount() {
        return locks.size();
    }

    /** Number of currently held write locks (approximate). */
    public int writeLockedCount() {
        int c = 0;
        for (ReentrantReadWriteLock l : locks.values()) {
            if (l.isWriteLocked()) c++;
        }
        return c;
    }
}
