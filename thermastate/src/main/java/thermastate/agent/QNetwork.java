/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Pointer;

/**
 * Java wrapper around the native Global_Q_network + RewardScalar.
 *
 * Usage:
 *   QNetwork qn = QNetwork.load("/path/to/AC_Q_Net.pt", "/path/to/AC_Q_Scalar.pt");
 *   float[][] cost = qn.predict(pdfs, dataSizes, rootFanouts, innerFanouts);
 *   qn.close();
 */
public final class QNetwork implements AutoCloseable {

    public static final int PDF_SIZE = 16384;
    public static final int INNER_FANOUT_SIZE = 256;
    public static final int OUTPUT_SIZE = 2;

    private final ThermaStateInference lib;
    private final Pointer handle;

    private QNetwork(ThermaStateInference lib, Pointer handle) {
        this.lib = lib;
        this.handle = handle;
    }

    /** Load model from file paths. Returns null if load fails. */
    public static QNetwork load(String modelPath, String scalarPath) {
        ThermaStateInference lib = ThermaStateInference.instance();
        if (lib == null) throw new RuntimeException("Native inference library not available");
        Pointer handle = lib.thermastate_load(modelPath, scalarPath);
        if (handle == null) {
            throw new RuntimeException("Failed to load native model: " + modelPath);
        }
        return new QNetwork(lib, handle);
    }

    /**
     * Batch predict — evaluate N candidates in a single native call.
     *
     * @param pdf          float[N][16384]  data distribution per candidate
     * @param dataSize     float[N]          data_size per candidate
     * @param rootFanout   float[N]          root fanout per candidate
     * @param innerFanout  float[N][256]     inner fanout matrix per candidate
     * @return float[N][2]  (memory_cost, get_cost) per candidate
     */
    public float[][] predict(float[][] pdf,
                             float[] dataSize,
                             float[] rootFanout,
                             float[][] innerFanout) {
        int n = pdf.length;
        if (n == 0) return new float[0][0];

        // Pack into contiguous arrays
        float[] packedPdf = new float[n * PDF_SIZE];
        float[] packedInner = new float[n * INNER_FANOUT_SIZE];
        for (int i = 0; i < n; i++) {
            System.arraycopy(pdf[i], 0, packedPdf, i * PDF_SIZE, PDF_SIZE);
            System.arraycopy(innerFanout[i], 0, packedInner, i * INNER_FANOUT_SIZE, INNER_FANOUT_SIZE);
        }

        float[] output = new float[n * OUTPUT_SIZE];
        int rc = lib.thermastate_predict(handle, packedPdf, dataSize, rootFanout,
                                          packedInner, n, output);
        if (rc != 0) {
            throw new RuntimeException("Native inference failed, rc=" + rc);
        }

        // Unpack output
        float[][] result = new float[n][OUTPUT_SIZE];
        for (int i = 0; i < n; i++) {
            result[i][0] = output[i * OUTPUT_SIZE];       // memory_cost
            result[i][1] = output[i * OUTPUT_SIZE + 1];    // get_cost
        }
        return result;
    }

    /** Single-candidate convenience method. */
    public float[] predictSingle(float[] pdf, float dataSize, float rootFanout, float[] innerFanout) {
        return predict(new float[][]{pdf},
                       new float[]{dataSize},
                       new float[]{rootFanout},
                       new float[][]{innerFanout})[0];
    }

    @Override
    public void close() {
        if (handle != null) {
            lib.thermastate_free(handle);
        }
    }
}
