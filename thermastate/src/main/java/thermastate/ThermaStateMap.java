/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.function.BiConsumer;

import thermastate.index.Configuration;
import thermastate.index.Index;
import thermastate.index.KeyEncoder;

/**
 * ThermaState learning-index-backed key-value map.
 *
 * Two-step lookup per the paper:
 *   1. index.get(encodedKey) → int bundleId
 *   2. block.getValue(bundleId) → V
 *
 * Keys are pre-encoded via {@link KeyEncoder#encode(int, int, long) ⟨kg, sid, uk⟩}.
 *
 * <p><b>Bootstrap phase:</b> When model paths are configured, the map starts in
 * bootstrap mode — writes are buffered in a TreeMap until the threshold is reached,
 * then a one-shot {@link Index#bulkLoad} builds the Agent-optimized tree. Subsequent
 * reads and writes go through the normal Index path.
 *
 * @param <V> value type
 */
public class ThermaStateMap<V> implements Serializable {

    private static final Configuration DEFAULT_CONF;

    static {
        DEFAULT_CONF = new Configuration();
        DEFAULT_CONF.rootFanOut = 16;
        for (int j = 0; j < Configuration.INNER_FANOUT_COLUMN; j++) {
            DEFAULT_CONF.fanOuts[0][j] = 4;
        }
    }

    /** Default bootstrap threshold — collect this many unique keys before bulkLoad. */
    public static final int DEFAULT_BOOTSTRAP_THRESHOLD = 1000;

    private final Index index;
    private ValueBundleBlock<V> block;

    // ── Bootstrap phase ──
    private boolean bootstrapComplete;
    private TreeMap<Double, V> bootstrapBuffer;
    private int bootstrapThreshold;

    // ── Model paths for Agent-driven bulkLoad ──
    private String globalModelPath;
    private String globalScalarPath;
    private String leafModelPath;

    public static long hashUserKey(Object key) {
        return ((long) key.hashCode()) & 0xFFFF_FFFFL;
    }

    public static <V> ThermaStateMap<V> create() {
        return create(DEFAULT_CONF);
    }

    public static <V> ThermaStateMap<V> create(Configuration conf) {
        return new ThermaStateMap<>(conf, -(double) Integer.MAX_VALUE, (double) Integer.MAX_VALUE);
    }

    public Configuration getConfiguration() {
        return index.getConfig();
    }

    public Index.IndexStats getStats() {
        finishBootstrap();
        return index.getStats();
    }

    public static Configuration defaultConfiguration() {
        return DEFAULT_CONF;
    }

    public ThermaStateMap(Configuration conf, double keyMin, double keyMax) {
        this.index = new Index(conf, keyMin, keyMax);
        this.block = new ValueBundleBlock<>();
        this.bootstrapComplete = true;   // no bootstrap by default
        this.bootstrapBuffer = null;
        this.bootstrapThreshold = 0;
    }

    // ── Bootstrap configuration ────────────────────────────────────────

    /**
     * Enable the cold-start bootstrap phase.
     * Writes are buffered until {@code threshold} unique keys are collected,
     * then a one-shot {@link Index#bulkLoad} builds the Agent-optimized tree.
     */
    public void enableBootstrap(int threshold) {
        this.bootstrapThreshold = threshold;
        this.bootstrapBuffer = new TreeMap<>();
        this.bootstrapComplete = false;
    }

    /** Set model paths for Agent-driven tree construction during bootstrap. */
    public void setModelPaths(String globalModelPath, String globalScalarPath, String leafModelPath) {
        this.globalModelPath = globalModelPath;
        this.globalScalarPath = globalScalarPath;
        this.leafModelPath = leafModelPath;
        index.setGlobalModelPath(globalModelPath, globalScalarPath);
        index.setModelPath(leafModelPath);
    }

    public String getGlobalModelPath() { return globalModelPath; }
    public String getLeafModelPath() { return leafModelPath; }
    public boolean isBootstrapComplete() { return bootstrapComplete; }
    public int getBootstrapBufferSize() {
        return bootstrapBuffer != null ? bootstrapBuffer.size() : 0;
    }

    // ── CRUD ───────────────────────────────────────────────────────────

    public V get(double encodedKey) {
        if (!bootstrapComplete) {
            return bootstrapBuffer.get(encodedKey);
        }
        int bundleId = index.get(encodedKey);
        if (bundleId < 0) return null;
        return block.getValue(bundleId);
    }

    public void put(double encodedKey, V value) {
        if (!bootstrapComplete) {
            bootstrapBuffer.put(encodedKey, value);
            if (bootstrapBuffer.size() >= bootstrapThreshold) {
                completeBootstrap();
            }
            return;
        }
        int bundleId = index.get(encodedKey);
        if (bundleId >= 0) {
            block.setValue(bundleId, value);
        } else {
            int newId = block.add(encodedKey, value);
            index.put(encodedKey, newId);
        }
    }

    public V remove(double encodedKey) {
        if (!bootstrapComplete) {
            return bootstrapBuffer.remove(encodedKey);
        }
        int bundleId = index.get(encodedKey);
        if (bundleId < 0) return null;
        V old = block.getValue(bundleId);
        block.setValue(bundleId, null);
        index.erase(encodedKey);
        return old;
    }

    public boolean containsKey(double encodedKey) {
        if (!bootstrapComplete) {
            return bootstrapBuffer.containsKey(encodedKey);
        }
        int bundleId = index.get(encodedKey);
        return bundleId >= 0 && block.getValue(bundleId) != null;
    }

    public void putAll(double[] encodedKeys, V[] values) {
        for (int i = 0; i < encodedKeys.length; i++) {
            put(encodedKeys[i], values[i]);
        }
    }

    public int size() {
        if (!bootstrapComplete) {
            return bootstrapBuffer.size();
        }
        return index.size();
    }

    public void forEach(BiConsumer<Double, V> action) {
        if (!bootstrapComplete) {
            for (Map.Entry<Double, V> e : bootstrapBuffer.entrySet()) {
                action.accept(e.getKey(), e.getValue());
            }
            return;
        }
        index.forEach((key, bundleId) -> {
            V val = block.getValue(bundleId);
            if (val != null) {
                action.accept(key, val);
            }
        });
    }

    public Index getIndex() {
        return index;
    }

    public ValueBundleBlock<V> getBlock() {
        return block;
    }

    // ── Bootstrap completion ───────────────────────────────────────────

    /**
     * Force-complete the bootstrap phase (no-op if already complete or not enabled).
     * Called automatically when the buffer reaches the threshold; call this
     * explicitly to finalize after a bounded data ingestion (e.g. end of a test).
     */
    public synchronized void finishBootstrap() {
        if (bootstrapComplete || bootstrapBuffer == null) return;
        completeBootstrap();
    }

    private synchronized void completeBootstrap() {
        if (bootstrapComplete || bootstrapBuffer == null || bootstrapBuffer.isEmpty()) {
            bootstrapComplete = true;
            bootstrapBuffer = null;
            return;
        }

        // Sort entries by encoded key
        List<Map.Entry<Double, V>> sorted = new ArrayList<>(bootstrapBuffer.entrySet());
        sorted.sort(Map.Entry.comparingByKey());

        // Assign bundleIds in sorted order, build IntEntry list for bulkLoad
        List<Index.IntEntry> entries = new ArrayList<>(sorted.size());
        for (Map.Entry<Double, V> e : sorted) {
            int bundleId = block.add(e.getKey(), e.getValue());
            entries.add(new Index.IntEntry(e.getKey(), bundleId));
        }

        // Agent-driven tree construction (Route Agent + TreePlan if model paths set)
        index.bulkLoad(entries);

        bootstrapBuffer.clear();
        bootstrapBuffer = null;
        bootstrapComplete = true;
    }
}
