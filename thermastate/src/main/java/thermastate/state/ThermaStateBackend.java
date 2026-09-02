/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.state;

import org.apache.flink.api.common.ExecutionConfig;
import org.apache.flink.api.common.JobID;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.core.fs.CloseableRegistry;
import org.apache.flink.metrics.MetricGroup;
import org.apache.flink.runtime.execution.Environment;
import org.apache.flink.runtime.query.TaskKvStateRegistry;
import org.apache.flink.runtime.state.*;
import org.apache.flink.runtime.state.metrics.LatencyTrackingStateConfig;
import org.apache.flink.runtime.state.ttl.TtlTimeProvider;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;

import thermastate.ThermaStateMap;
import thermastate.config.ConfigLoader;
import thermastate.index.Configuration;

/**
 * ThermaState StateBackend — uses the HITS learned index for keyed state storage.
 *
 * <p>Model auto-detection: looks for C++ trained models under
 * {@code ../ThermaIndex/data/model/} relative to the working directory.
 * Override with system properties:
 * <ul>
 *   <li>{@code thermastate.global.model.path}  — Route Agent Q-Network</li>
 *   <li>{@code thermastate.global.scalar.path}  — Route Agent reward scalar</li>
 *   <li>{@code thermastate.leaf.model.path}     — LeafAgent / TreePlan network</li>
 *   <li>{@code thermastate.bootstrap.threshold} — keys to collect before bulkLoad (default 1000)</li>
 * </ul>
 *
 * <p>When model paths are found, the backend enables a cold-start bootstrap phase:
 * data is buffered, then a one-shot {@code Index.bulkLoad} builds the Agent-optimized
 * tree before switching to online mode.
 */
public class ThermaStateBackend extends AbstractStateBackend {

    private static final String CONFIG_FILE = "thermastate_config.json";

    // System property keys
    public static final String GLOBAL_MODEL_PATH_PROP  = "thermastate.global.model.path";
    public static final String GLOBAL_SCALAR_PATH_PROP = "thermastate.global.scalar.path";
    public static final String LEAF_MODEL_PATH_PROP    = "thermastate.leaf.model.path";
    public static final String BOOTSTRAP_THRESHOLD_PROP = "thermastate.bootstrap.threshold";

    // Default search paths (relative to working directory)
    private static final String[] SEARCH_ROOTS = { ".", ".." };
    private static final String RELATIVE_GLOBAL_MODEL  = "ThermaIndex/data/model/AC_Q_Net.pt";
    private static final String RELATIVE_GLOBAL_SCALAR = "ThermaIndex/data/model/AC_Q_Scalar.pt";
    private static final String RELATIVE_LEAF_MODEL    = "ThermaIndex/data/model/leaf_agent.pt";

    private final Configuration indexConfig;
    private final String globalModelPath;
    private final String globalScalarPath;
    private final String leafModelPath;
    private final int bootstrapThreshold;

    private static volatile ThermaKeyedStateBackend<?> lastCreated;

    /** Auto-detect: load config file if present, scan for model files. */
    public ThermaStateBackend() {
        this(autoDetectConfig(), detectModelPath(GLOBAL_MODEL_PATH_PROP, RELATIVE_GLOBAL_MODEL),
             detectModelPath(GLOBAL_SCALAR_PATH_PROP, RELATIVE_GLOBAL_SCALAR),
             detectModelPath(LEAF_MODEL_PATH_PROP, RELATIVE_LEAF_MODEL),
             detectBootstrapThreshold());
    }

    /** Create with specific config and model paths. */
    public ThermaStateBackend(Configuration indexConfig,
                              String globalModelPath, String globalScalarPath,
                              String leafModelPath, int bootstrapThreshold) {
        this.indexConfig = indexConfig;
        this.globalModelPath = globalModelPath;
        this.globalScalarPath = globalScalarPath;
        this.leafModelPath = leafModelPath;
        this.bootstrapThreshold = bootstrapThreshold;
    }

    /** Create with specific index config, no models. */
    public ThermaStateBackend(Configuration indexConfig) {
        this(indexConfig, null, null, null, 0);
    }

    // ── Model path detection ─────────────────────────────────────────

    private static String detectModelPath(String propKey, String relativePath) {
        // 1. System property wins
        String prop = System.getProperty(propKey);
        if (prop != null && !prop.isEmpty()) {
            File f = new File(prop);
            if (f.exists()) {
                System.out.println("[ThermaState] model from system property " + propKey + "=" + f.getAbsolutePath());
                return f.getAbsolutePath();
            }
            System.err.println("[ThermaState] WARNING: " + propKey + "=" + prop + " not found, scanning...");
        }

        // 2. Scan search roots
        for (String root : SEARCH_ROOTS) {
            File f = new File(root, relativePath);
            if (f.exists()) {
                System.out.println("[ThermaState] found model: " + f.getAbsolutePath());
                return f.getAbsolutePath();
            }
        }

        System.out.println("[ThermaState] model not found: " + relativePath + " (scanned " + SEARCH_ROOTS.length + " roots)");
        return null;
    }

    private static int detectBootstrapThreshold() {
        String prop = System.getProperty(BOOTSTRAP_THRESHOLD_PROP);
        if (prop != null && !prop.isEmpty()) {
            try {
                return Integer.parseInt(prop);
            } catch (NumberFormatException e) {
                System.err.println("[ThermaState] invalid bootstrap threshold: " + prop);
            }
        }
        return ThermaStateMap.DEFAULT_BOOTSTRAP_THRESHOLD;
    }

    // ── Config ───────────────────────────────────────────────────────

    private static Configuration autoDetectConfig() {
        File f = new File(CONFIG_FILE);
        if (f.exists()) {
            try {
                Configuration conf = ConfigLoader.load(f.getAbsolutePath());
                System.out.println("[ThermaState] loaded config from: " + f.getAbsolutePath());
                return conf;
            } catch (IOException e) {
                System.err.println("[ThermaState] failed to load config, using defaults: " + e.getMessage());
            }
        }
        Configuration conf = new Configuration();
        conf.rootFanOut = 64;
        for (int j = 0; j < Configuration.INNER_FANOUT_COLUMN; j++) conf.fanOuts[0][j] = 8;
        System.out.println("[ThermaState] using default config (rootFanOut=64, innerFanout=8)");
        return conf;
    }

    // ── StateBackend contract ────────────────────────────────────────

    @Override
    public <K> AbstractKeyedStateBackend<K> createKeyedStateBackend(
            Environment env,
            JobID jobID,
            String operatorIdentifier,
            TypeSerializer<K> keySerializer,
            int numberOfKeyGroups,
            KeyGroupRange keyGroupRange,
            TaskKvStateRegistry kvStateRegistry,
            TtlTimeProvider ttlTimeProvider,
            MetricGroup metricGroup,
            Collection<KeyedStateHandle> stateHandles,
            CloseableRegistry cancelStreamRegistry) throws IOException {

        ExecutionConfig execConfig = env.getExecutionConfig();
        ClassLoader cl = env.getUserCodeClassLoader().asClassLoader();

        ThermaKeyedStateBackend<K> backend = new ThermaKeyedStateBackend<>(
                kvStateRegistry,
                keySerializer,
                cl,
                execConfig,
                ttlTimeProvider,
                LatencyTrackingStateConfig.disabled(),
                cancelStreamRegistry,
                keyGroupRange,
                numberOfKeyGroups,
                stateHandles,
                indexConfig,
                globalModelPath, globalScalarPath, leafModelPath,
                bootstrapThreshold);
        lastCreated = backend;
        return backend;
    }

    @Override
    public OperatorStateBackend createOperatorStateBackend(
            Environment env,
            String operatorIdentifier,
            Collection<OperatorStateHandle> stateHandles,
            CloseableRegistry cancelStreamRegistry) {

        return new DefaultOperatorStateBackend(
                env.getExecutionConfig(),
                cancelStreamRegistry,
                new HashMap<>(),
                new HashMap<>(),
                new HashMap<>(),
                new HashMap<>(),
                new SnapshotStrategyRunner<>(
                    "thermastate-operator",
                    null,
                    cancelStreamRegistry,
                    SnapshotExecutionType.ASYNCHRONOUS));
    }

    // ── Diagnostics ──────────────────────────────────────────────────

    /** Expose the state store of the last created backend (for tests/diagnostics). */
    public static ThermaStateMap<Object> getStateStore() {
        ThermaKeyedStateBackend<?> b = lastCreated;
        if (b != null) {
            ThermaStateMap<Object> store = b.getStateStore();
            if (store != null) store.finishBootstrap();
            return store;
        }
        return null;
    }

    public boolean hasGlobalModel() { return globalModelPath != null; }
    public boolean hasLeafModel() { return leafModelPath != null; }
    public String getGlobalModelPath() { return globalModelPath; }
    public String getLeafModelPath() { return leafModelPath; }
    public int getBootstrapThreshold() { return bootstrapThreshold; }
}
