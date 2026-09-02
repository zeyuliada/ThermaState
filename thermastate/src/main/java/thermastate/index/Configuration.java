/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

import java.io.Serializable;
import java.util.Random;

public class Configuration implements Serializable {
    public static final int INNER_FANOUT_ROW = 1;
    public static final int INNER_FANOUT_COLUMN = 256;
    public static final int INNER_FANOUT_SIZE = INNER_FANOUT_ROW * INNER_FANOUT_COLUMN;

    public static final float DEFAULT_ROOT_FANOUT = 333.0f;
    public static final float DEFAULT_INNER_FANOUT = 10.0f;

    public static final float MIN_ROOT_FANOUT = 1.0f;
    public static final float MAX_ROOT_FANOUT = 256.0f * 1024.0f;
    public static final float MIN_INNER_FANOUT = 1.0f;
    public static final float MAX_INNER_FANOUT = 1024.0f;

    public float rootFanOut;
    public float[][] fanOuts;
    /** Per-root-bucket inner fanout decisions (TS-MDP / DARE output).
     *  Length = rootFanOut. Null means "not populated, use heuristic or decideFanout". */
    public int[] innerFanouts;

    public Configuration() {
        this.rootFanOut = DEFAULT_ROOT_FANOUT;
        this.fanOuts = new float[INNER_FANOUT_ROW][INNER_FANOUT_COLUMN];
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                this.fanOuts[i][j] = DEFAULT_INNER_FANOUT;
            }
        }
    }

    /** Get inner fanout for a root bucket slot; falls back to 8 if not configured. */
    public int getInnerFanout(int rootSlot) {
        if (innerFanouts != null && rootSlot < innerFanouts.length) {
            int f = innerFanouts[rootSlot];
            return f > 0 ? f : 8;
        }
        return 8;
    }

    public static Configuration defaultConfiguration() {
        Configuration conf = new Configuration();
        conf.rootFanOut = 30000;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                conf.fanOuts[i][j] = 100;
            }
        }
        return conf;
    }

    public void shrink() {
        rootFanOut = clamp(rootFanOut, MIN_ROOT_FANOUT, MAX_ROOT_FANOUT);
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                fanOuts[i][j] = clamp(fanOuts[i][j], MIN_INNER_FANOUT, MAX_INNER_FANOUT);
            }
        }
    }

    public static Configuration zeros() {
        Configuration conf = new Configuration();
        conf.rootFanOut = 1;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                conf.fanOuts[i][j] = 1e-6f;
            }
        }
        return conf;
    }

    public static Configuration randomConfiguration(Random rng) {
        Configuration conf = new Configuration();
        conf.rootFanOut = (float) (Math.pow(rng.nextDouble(), 2) * (MAX_ROOT_FANOUT - MIN_ROOT_FANOUT) + MIN_ROOT_FANOUT);
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                conf.fanOuts[i][j] = (float) (Math.pow(rng.nextDouble(), 2) * (MAX_INNER_FANOUT - MIN_INNER_FANOUT) + MIN_INNER_FANOUT);
            }
        }
        return conf;
    }

    public Configuration add(Configuration other) {
        Configuration result = new Configuration();
        result.rootFanOut = this.rootFanOut + other.rootFanOut;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                result.fanOuts[i][j] = this.fanOuts[i][j] + other.fanOuts[i][j];
            }
        }
        return result;
    }

    public Configuration multiply(float scalar) {
        Configuration result = new Configuration();
        result.rootFanOut = this.rootFanOut * scalar;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                result.fanOuts[i][j] = this.fanOuts[i][j] * scalar;
            }
        }
        return result;
    }

    public Configuration subtract(Configuration other) {
        Configuration result = new Configuration();
        result.rootFanOut = this.rootFanOut - other.rootFanOut;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                result.fanOuts[i][j] = this.fanOuts[i][j] - other.fanOuts[i][j];
            }
        }
        return result;
    }

    public Configuration multiplyElements(Configuration other) {
        Configuration result = new Configuration();
        result.rootFanOut = this.rootFanOut * other.rootFanOut;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                result.fanOuts[i][j] = this.fanOuts[i][j] * other.fanOuts[i][j];
            }
        }
        return result;
    }

    public Configuration divide(float scalar) {
        Configuration result = new Configuration();
        result.rootFanOut = this.rootFanOut / scalar;
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                result.fanOuts[i][j] = this.fanOuts[i][j] / scalar;
            }
        }
        return result;
    }

    private static float clamp(float val, float min, float max) {
        return Math.max(min, Math.min(max, val));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("root:").append(rootFanOut).append("\n");
        for (int i = 0; i < INNER_FANOUT_ROW; i++) {
            sb.append(i).append("{");
            for (int j = 0; j < INNER_FANOUT_COLUMN; j++) {
                sb.append(fanOuts[i][j]).append(" ");
            }
            sb.append("}\n");
        }
        return sb.toString();
    }
}
