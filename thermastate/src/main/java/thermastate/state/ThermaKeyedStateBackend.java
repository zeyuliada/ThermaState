/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.state;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.state.State;
import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.state.AggregatingStateDescriptor;
import org.apache.flink.api.common.state.StateDescriptor;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.core.fs.CloseableRegistry;
import org.apache.flink.runtime.checkpoint.CheckpointOptions;
import org.apache.flink.runtime.query.TaskKvStateRegistry;
import org.apache.flink.runtime.state.*;
import org.apache.flink.runtime.state.heap.HeapPriorityQueueElement;
import org.apache.flink.runtime.state.heap.HeapPriorityQueueSet;
import org.apache.flink.runtime.state.heap.InternalKeyContext;
import org.apache.flink.runtime.state.PriorityComparator;
import org.apache.flink.runtime.state.KeyExtractorFunction;
import org.apache.flink.runtime.state.Keyed;
import org.apache.flink.runtime.state.internal.InternalValueState;
import org.apache.flink.runtime.state.metrics.LatencyTrackingStateConfig;
import org.apache.flink.runtime.state.ttl.TtlTimeProvider;
import org.apache.flink.runtime.util.NonClosingInputStreamDecorator;
import org.apache.flink.runtime.util.NonClosingOutputStreamDecorator;

import thermastate.ThermaStateMap;
import thermastate.index.Configuration;
import thermastate.index.KeyEncoder;

import java.io.*;
import java.util.*;
import java.util.concurrent.RunnableFuture;
import java.util.stream.Stream;

/**
 * ThermaState KeyedStateBackend — keyed state stored in a ThermaStateMap
 * backed by the HITS learned index.
 */
public class ThermaKeyedStateBackend<K> extends AbstractKeyedStateBackend<K> {

    private static final StreamCompressionDecorator NOOP_COMPRESSION =
        new StreamCompressionDecorator() {
            @Override
            protected OutputStream decorateWithCompression(
                    NonClosingOutputStreamDecorator stream) {
                return stream;
            }
            @Override
            protected InputStream decorateWithCompression(
                    NonClosingInputStreamDecorator stream) {
                return stream;
            }
        };

    private final ThermaStateMap<Object> stateStore;
    private final Map<String, InternalValueState<K, VoidNamespace, ?>> states;
    private final Map<String, Integer> stateIds;
    private int nextStateId;
    private final Configuration indexConfiguration;

    @SuppressWarnings("unchecked")
    public ThermaKeyedStateBackend(
            TaskKvStateRegistry kvStateRegistry,
            TypeSerializer<K> keySerializer,
            ClassLoader userCodeClassLoader,
            ExecutionConfig executionConfig,
            TtlTimeProvider ttlTimeProvider,
            LatencyTrackingStateConfig latencyTrackingStateConfig,
            CloseableRegistry cancelStreamRegistry,
            KeyGroupRange keyGroupRange,
            int numberOfKeyGroups,
            Collection<KeyedStateHandle> stateHandles) {
        this(kvStateRegistry, keySerializer, userCodeClassLoader, executionConfig,
             ttlTimeProvider, latencyTrackingStateConfig, cancelStreamRegistry,
             keyGroupRange, numberOfKeyGroups, stateHandles, null);
    }

    @SuppressWarnings("unchecked")
    public ThermaKeyedStateBackend(
            TaskKvStateRegistry kvStateRegistry,
            TypeSerializer<K> keySerializer,
            ClassLoader userCodeClassLoader,
            ExecutionConfig executionConfig,
            TtlTimeProvider ttlTimeProvider,
            LatencyTrackingStateConfig latencyTrackingStateConfig,
            CloseableRegistry cancelStreamRegistry,
            KeyGroupRange keyGroupRange,
            int numberOfKeyGroups,
            Collection<KeyedStateHandle> stateHandles,
            Configuration indexConfig) {
        this(kvStateRegistry, keySerializer, userCodeClassLoader, executionConfig,
             ttlTimeProvider, latencyTrackingStateConfig, cancelStreamRegistry,
             keyGroupRange, numberOfKeyGroups, stateHandles, indexConfig,
             null, null, null, 0);
    }

    @SuppressWarnings("unchecked")
    public ThermaKeyedStateBackend(
            TaskKvStateRegistry kvStateRegistry,
            TypeSerializer<K> keySerializer,
            ClassLoader userCodeClassLoader,
            ExecutionConfig executionConfig,
            TtlTimeProvider ttlTimeProvider,
            LatencyTrackingStateConfig latencyTrackingStateConfig,
            CloseableRegistry cancelStreamRegistry,
            KeyGroupRange keyGroupRange,
            int numberOfKeyGroups,
            Collection<KeyedStateHandle> stateHandles,
            Configuration indexConfig,
            String globalModelPath, String globalScalarPath,
            String leafModelPath, int bootstrapThreshold) {

        super(kvStateRegistry, keySerializer, userCodeClassLoader, executionConfig,
              ttlTimeProvider, latencyTrackingStateConfig, cancelStreamRegistry,
              NOOP_COMPRESSION, new ThermaKeyContext(keyGroupRange, numberOfKeyGroups));

        this.indexConfiguration = (indexConfig != null) ? indexConfig : ThermaStateMap.defaultConfiguration();
        this.stateStore = ThermaStateMap.create(this.indexConfiguration);
        this.states = new HashMap<>();
        this.stateIds = new HashMap<>();
        this.nextStateId = 0;

        // Wire model paths + bootstrap if configured
        boolean hasGlobal = globalModelPath != null && !globalModelPath.isEmpty()
                         && globalScalarPath != null && !globalScalarPath.isEmpty();
        boolean hasLeaf = leafModelPath != null && !leafModelPath.isEmpty();

        if (hasGlobal || hasLeaf) {
            stateStore.setModelPaths(
                hasGlobal ? globalModelPath : null,
                hasGlobal ? globalScalarPath : null,
                hasLeaf ? leafModelPath : null);
        }

        if (bootstrapThreshold > 0) {
            stateStore.enableBootstrap(bootstrapThreshold);
            System.out.println("[ThermaState] bootstrap enabled: threshold=" + bootstrapThreshold
                + " routeAgent=" + hasGlobal + " treePlan=" + hasLeaf);
        }
    }

    /** The Configuration used to build the underlying Index. */
    public Configuration getIndexConfiguration() {
        return indexConfiguration;
    }

    ThermaStateMap<Object> getStateStore() {
        return stateStore;
    }

    /** Allocate or look up a stable sid for a state descriptor name. */
    int getStateId(String stateName) {
        return stateIds.computeIfAbsent(stateName, k -> nextStateId++);
    }

    /** Encode ⟨kg, sid, uk⟩ for the current key. */
    double encodeKey(int sid, K key) {
        int kg = getCurrentKeyGroupIndex();
        long uk = ThermaStateMap.hashUserKey(key);
        return KeyEncoder.encode(kg, sid, uk);
    }

    // -- PriorityQueueSetFactory --

    @SuppressWarnings("unchecked")
    @Override
    public <T extends HeapPriorityQueueElement & PriorityComparable<? super T> & Keyed<?>>
            KeyGroupedInternalPriorityQueue<T> create(String stateName, TypeSerializer<T> typeSerializer) {
        return new HeapPriorityQueueSet<>(
                PriorityComparator.forPriorityComparableObjects(),
                KeyExtractorFunction.forKeyedObjects(),
                128,
                getKeyGroupRange(),
                getNumberOfKeyGroups());
    }

    // -- State creation --

    @SuppressWarnings("unchecked")
    @Override
    public <N, SV, SEV, S extends State, IS extends S> IS createOrUpdateInternalState(
            TypeSerializer<N> namespaceSerializer,
            StateDescriptor<S, SV> stateDesc,
            StateSnapshotTransformer.StateSnapshotTransformFactory<SEV> snapshotTransform)
            throws Exception {

        String name = stateDesc.getName();

        if (stateDesc.getType() == StateDescriptor.Type.AGGREGATING) {
            @SuppressWarnings("rawtypes")
            AggregatingStateDescriptor<?, ?, ?> aggDesc =
                (AggregatingStateDescriptor<?, ?, ?>) stateDesc;
            AggregateFunction<?, ?, ?> aggFn = aggDesc.getAggregateFunction();
            ThermaAggregatingState<K, N, ?, ?, ?> state =
                new ThermaAggregatingState<>(this, stateDesc, aggFn);
            @SuppressWarnings({"unchecked", "rawtypes"})
            IS result = (IS) state;
            return result;
        }

        if (stateDesc.getType() != StateDescriptor.Type.VALUE) {
            throw new UnsupportedOperationException(
                "ThermaState only supports ValueState and AggregatingState. Got: "
                    + stateDesc.getType());
        }

        if (states.containsKey(name)) {
            return (IS) states.get(name);
        }

        ThermaValueState<K, N, SV> state = new ThermaValueState<>(this, stateDesc);
        states.put(name, (InternalValueState<K, VoidNamespace, ?>) state);
        publishQueryableStateIfEnabled(stateDesc, state);
        return (IS) state;
    }

    // -- Key iteration --

    @Override
    public <N> Stream<K> getKeys(String state, N namespace) {
        return Stream.empty();
    }

    @Override
    public <N> Stream<Tuple2<K, N>> getKeysAndNamespaces(String state) {
        return Stream.empty();
    }

    @Override
    public int numKeyValueStateEntries() {
        return stateStore.size();
    }

    // -- Snapshot --

    @Override
    public boolean isSafeToReuseKVState() {
        return true;
    }

    @Override
    public RunnableFuture<SnapshotResult<KeyedStateHandle>> snapshot(
            long checkpointId, long timestamp,
            CheckpointStreamFactory streamFactory,
            CheckpointOptions checkpointOptions) {
        return null;
    }

    @Override
    public SavepointResources<K> savepoint() {
        return null;
    }

    @Override
    public void notifyCheckpointComplete(long checkpointId) {
    }

    @Override
    public void notifyCheckpointAborted(long checkpointId) {
    }

    // -- InternalKeyContext delegating impl --

    private static class ThermaKeyContext<K> implements InternalKeyContext<K> {
        private final KeyGroupRange keyGroupRange;
        private final int numberOfKeyGroups;
        private K currentKey;
        private int currentKeyGroupIndex;

        ThermaKeyContext(KeyGroupRange keyGroupRange, int numberOfKeyGroups) {
            this.keyGroupRange = keyGroupRange;
            this.numberOfKeyGroups = numberOfKeyGroups;
        }

        @Override public K getCurrentKey() { return currentKey; }
        @Override public int getCurrentKeyGroupIndex() { return currentKeyGroupIndex; }
        @Override public int getNumberOfKeyGroups() { return numberOfKeyGroups; }
        @Override public KeyGroupRange getKeyGroupRange() { return keyGroupRange; }
        @Override public void setCurrentKey(K key) { this.currentKey = key; }
        @Override public void setCurrentKeyGroupIndex(int idx) { this.currentKeyGroupIndex = idx; }
    }
}
