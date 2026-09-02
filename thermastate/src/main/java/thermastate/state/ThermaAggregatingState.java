/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.state;

import org.apache.flink.api.common.functions.AggregateFunction;
import org.apache.flink.api.common.state.StateDescriptor;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.runtime.state.internal.InternalAggregatingState;
import org.apache.flink.runtime.state.internal.InternalKvState;

import thermastate.ThermaStateMap;

import java.util.Collection;

/**
 * AggregatingState backed by ThermaStateMap — one accumulator per key+namespace.
 * Needed by Flink's window operator when using AggregateFunction.
 */
public class ThermaAggregatingState<K, N, IN, ACC, OUT>
        implements InternalAggregatingState<K, N, IN, ACC, OUT> {

    private final ThermaKeyedStateBackend<K> backend;
    private final StateDescriptor<?, ACC> stateDesc;
    private final AggregateFunction<IN, ACC, OUT> aggregateFunction;
    private final int sid;
    private N currentNamespace;
    private boolean cleared;

    @SuppressWarnings("unchecked")
    ThermaAggregatingState(ThermaKeyedStateBackend<K> backend,
                           StateDescriptor<?, ?> stateDesc,
                           AggregateFunction<?, ?, ?> aggregateFunction) {
        this.backend = backend;
        this.stateDesc = (StateDescriptor<?, ACC>) stateDesc;
        this.aggregateFunction = (AggregateFunction<IN, ACC, OUT>) aggregateFunction;
        this.sid = backend.getStateId(stateDesc.getName());
        this.cleared = false;
    }

    // -- InternalKvState --

    @Override
    public void setCurrentNamespace(N namespace) {
        this.currentNamespace = namespace;
    }

    @Override
    public TypeSerializer<K> getKeySerializer() {
        return backend.getKeySerializer();
    }

    @Override
    public TypeSerializer<N> getNamespaceSerializer() {
        throw new UnsupportedOperationException("namespace serializer not available");
    }

    @SuppressWarnings("unchecked")
    @Override
    public TypeSerializer<ACC> getValueSerializer() {
        return (TypeSerializer<ACC>) stateDesc.getSerializer();
    }

    @Override
    public byte[] getSerializedValue(
            byte[] serializedKeyAndNamespace,
            TypeSerializer<K> safeKeySerializer,
            TypeSerializer<N> safeNamespaceSerializer,
            TypeSerializer<ACC> safeValueSerializer) {
        throw new UnsupportedOperationException("queryable state not supported");
    }

    @Override
    public InternalKvState.StateIncrementalVisitor<K, N, ACC> getStateIncrementalVisitor(
            int recommendedMaxNumberOfReturnedRecords) {
        throw new UnsupportedOperationException("state iteration not supported");
    }

    // -- AppendingState --

    @Override
    public OUT get() {
        ACC acc = getInternal();
        if (acc == null) return null;
        return aggregateFunction.getResult(acc);
    }

    @Override
    public void add(IN value) throws Exception {
        ACC current = getInternal();
        if (current == null) {
            current = aggregateFunction.createAccumulator();
        }
        ACC updated = aggregateFunction.add(value, current);
        updateInternal(updated);
    }

    // -- InternalAppendingState --

    @Override
    @SuppressWarnings("unchecked")
    public ACC getInternal() {
        if (cleared) return null;
        K key = backend.getCurrentKey();
        double encoded = backend.encodeKey(sid, key);
        return (ACC) backend.getStateStore().get(encoded);
    }

    @Override
    public void updateInternal(ACC value) {
        K key = backend.getCurrentKey();
        double encoded = backend.encodeKey(sid, key);
        backend.getStateStore().put(encoded, value);
        cleared = false;
    }

    // -- InternalMergingState --

    @Override
    public void mergeNamespaces(N target, Collection<N> sources) throws Exception {
        throw new UnsupportedOperationException("merge not supported");
    }

    @Override
    public void clear() {
        K key = backend.getCurrentKey();
        double encoded = backend.encodeKey(sid, key);
        backend.getStateStore().remove(encoded);
        cleared = true;
    }

    @Override
    public String toString() {
        return "ThermaAggregatingState{name=" + stateDesc.getName() + "}";
    }
}
