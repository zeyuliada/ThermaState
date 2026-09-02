/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.state;

import org.apache.flink.api.common.state.StateDescriptor;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.runtime.state.internal.InternalValueState;

/**
 * ValueState backed by ThermaStateMap (HITS learned index).
 *
 * Each state instance stores values for all keys in a single ThermaStateMap,
 * keyed by composite string: "{stateName}|{namespace}|{key}".
 */
public class ThermaValueState<K, N, V> implements InternalValueState<K, N, V> {

    private final ThermaKeyedStateBackend<K> backend;
    private final StateDescriptor<?, V> stateDesc;
    private final int sid;
    private N currentNamespace;
    private boolean cleared;

    ThermaValueState(ThermaKeyedStateBackend<K> backend, StateDescriptor<?, V> stateDesc) {
        this.backend = backend;
        this.stateDesc = stateDesc;
        this.sid = backend.getStateId(stateDesc.getName());
        this.cleared = false;
    }

    // --- InternalKvState ---

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
    public TypeSerializer<V> getValueSerializer() {
        return (TypeSerializer<V>) stateDesc.getSerializer();
    }

    @Override
    public byte[] getSerializedValue(
            byte[] serializedKeyAndNamespace,
            TypeSerializer<K> safeKeySerializer,
            TypeSerializer<N> safeNamespaceSerializer,
            TypeSerializer<V> safeValueSerializer) {
        throw new UnsupportedOperationException("queryable state not supported");
    }

    @Override
    public StateIncrementalVisitor<K, N, V> getStateIncrementalVisitor(
            int recommendedMaxNumberOfReturnedRecords) {
        throw new UnsupportedOperationException("state iteration not supported");
    }

    // --- ValueState ---

    @Override
    @SuppressWarnings("unchecked")
    public V value() {
        if (cleared) return null;
        K key = backend.getCurrentKey();
        double encoded = backend.encodeKey(sid, key);
        return (V) backend.getStateStore().get(encoded);
    }

    @Override
    public void update(V value) {
        if (value == null) {
            clear();
            return;
        }
        K key = backend.getCurrentKey();
        double encoded = backend.encodeKey(sid, key);
        backend.getStateStore().put(encoded, value);
        cleared = false;
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
        return "ThermaValueState{name=" + stateDesc.getName() + "}";
    }
}
