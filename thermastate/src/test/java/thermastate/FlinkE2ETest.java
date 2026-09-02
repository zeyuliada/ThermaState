package thermastate;

import static org.junit.Assert.*;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import org.apache.flink.util.Collector;

import org.junit.Test;

import thermastate.state.ThermaStateBackend;

/**
 * End-to-end Flink pipeline test with ThermaStateBackend.
 *
 * Runs a simplified WordCount: source emits keyId:word pairs,
 * StatefulProcessor accumulates word counts in ValueState,
 * sink collects results for verification.
 *
 * No external Flink cluster required — uses local MiniCluster with parallelism 2.
 */
public class FlinkE2ETest {

    private static final int NUM_KEYS = 10_000;
    private static final int NUM_ROUNDS = 10;
    private static final int NUM_DISTINCT_WORDS = 100;

    private static final List<String> sinkResults = new CopyOnWriteArrayList<>();

    // ── Source ─────────────────────────────────────────────────────────

    public static class TestSource extends RichParallelSourceFunction<String> {
        private volatile boolean running = true;

        @Override
        public void run(SourceContext<String> ctx) throws Exception {
            long ts = System.currentTimeMillis();
            for (int round = 0; round < NUM_ROUNDS && running; round++) {
                for (int i = 0; i < NUM_KEYS; i++) {
                    String word = "w" + (i % NUM_DISTINCT_WORDS);
                    ctx.collectWithTimestamp(i + ":" + word, ts);
                    ts += 1;
                }
                Thread.sleep(100);
            }
            Thread.sleep(1000);
        }

        @Override
        public void cancel() { running = false; }
    }

    // ── Processor ──────────────────────────────────────────────────────

    public static class WordCountProcessor extends RichFlatMapFunction<String, String> {
        private transient ValueState<Integer> countState;

        @Override
        public void open(Configuration config) {
            ValueStateDescriptor<Integer> desc = new ValueStateDescriptor<>(
                "wordCount", Integer.class);
            countState = getRuntimeContext().getState(desc);
        }

        @Override
        public void flatMap(String record, Collector<String> out) throws Exception {
            String[] parts = record.split(":");
            if (parts.length < 2) return;
            String word = parts[1];

            Integer cur = countState.value();
            int next = (cur == null) ? 1 : cur + 1;
            countState.update(next);

            out.collect(word + "=" + next);
        }
    }

    // ── Sink ───────────────────────────────────────────────────────────

    public static class TestSink implements SinkFunction<String> {
        @Override
        public void invoke(String value, Context context) {
            sinkResults.add(value);
        }
    }

    // ── Test ───────────────────────────────────────────────────────────

    @Test
    public void testFlinkE2E() throws Exception {
        sinkResults.clear();

        // 1. Create Flink pipeline with ThermaStateBackend
        StreamExecutionEnvironment env =
            StreamExecutionEnvironment.createLocalEnvironment(2);
        env.setParallelism(2);
        env.setStateBackend(new ThermaStateBackend());

        DataStream<String> stream = env
            .addSource(new TestSource()).setParallelism(1)
            .keyBy(x -> x.split(":")[1])   // key by word, not keyId
            .flatMap(new WordCountProcessor()).setParallelism(2);

        stream.addSink(new TestSink()).setParallelism(1);

        System.out.println("=== FlinkE2ETest: starting pipeline ===");
        env.execute("FlinkE2ETest");
        System.out.println("=== FlinkE2ETest: pipeline complete ===");

        // 2. Verify results
        assertFalse("sink should have collected results", sinkResults.isEmpty());

        // Each of the NUM_DISTINCT_WORDS should have final count = NUM_KEYS / NUM_DISTINCT_WORDS * NUM_ROUNDS
        int expected = (NUM_KEYS / NUM_DISTINCT_WORDS) * NUM_ROUNDS;
        int maxCount = 0;
        for (String r : sinkResults) {
            String[] parts = r.split("=");
            if (parts.length == 2) {
                int count = Integer.parseInt(parts[1]);
                maxCount = Math.max(maxCount, count);
            }
        }
        assertEquals("max word count should match expected rounds", expected, maxCount);

        // 3. Verify ThermaStateBackend metrics
        ThermaStateMap<?> stateStore = ThermaStateBackend.getStateStore();
        assertNotNull("stateStore should be available after pipeline", stateStore);

        System.out.println("=== FlinkE2ETest PASSED ===");
        System.out.println("   Records in sink: " + sinkResults.size());
        System.out.println("   Max word count: " + maxCount + " (expected " + expected + ")");
    }
}
