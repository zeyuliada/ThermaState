/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.agent;

import com.sun.jna.Pointer;

/**
 * Route Agent — Global_Q_network GA search + DARE interpolation + temperature labeling.
 *
 * Flow:
 *   1. GA search via JNA thermastate_search → (rootFanout, fan_outs[256])
 *   2. DARE interpolation: fan_outs[pos_norm] → per-node fanout
 *   3. Temperature labeling: PDF density → T_ij → Hot/Cold
 *
 * All NN inference happens in C++ via JNA. Java only consumes the result.
 */
public final class RouteAgent {

    public static final int GLOBAL_PDF_SIZE = QNetwork.PDF_SIZE;     // 16384
    public static final int FANOUT_TABLE_SIZE = 256;                 // INNER_FANOUT_COLUMN

    // Temperature parameters (论文公式 8-9)
    private static final double BETA = 0.5;    // range correction weight
    private static final double TAU  = 0.3;    // hot/cold threshold

    private RouteAgent() {}

    // ── Result types ──

    /** Output of GA search via Q-Network. */
    public static final class RouteConfig {
        public final int rootFanout;
        public final float[] fanOuts;   // [256] non-uniform interpolation table

        RouteConfig(int rootFanout, float[] fanOuts) {
            this.rootFanout = rootFanout;
            this.fanOuts = fanOuts;
        }
    }

    /** Per-region descriptor for temperature labeling. */
    public static final class Region {
        public final double lo;
        public final double hi;
        double freq;       // filled by label()
        boolean isHot;     // filled by label()
        int fanout;        // filled by caller (DARE interpolation)

        public Region(double lo, double hi) {
            this.lo = lo;
            this.hi = hi;
        }
    }

    // ── GA Search ──

    /**
     * Run GA search via C++ Global_Q_network to find optimal Configuration.
     *
     * @param lib       JNA inference library
     * @param handle    opaque handle from thermastate_load()
     * @param globalPdf float[16384] data distribution
     * @param dataSize  total number of records
     */
    public static RouteConfig search(ThermaStateInference lib, Pointer handle,
                                      float[] globalPdf, float dataSize) {
        float[] outRoot = new float[1];
        float[] outInner = new float[FANOUT_TABLE_SIZE];
        int rc = lib.thermastate_search(handle, globalPdf, dataSize, outRoot, outInner);
        if (rc != 0) {
            throw new RuntimeException("thermastate_search failed, rc=" + rc);
        }
        return new RouteConfig((int) outRoot[0], outInner);
    }

    // ── DARE Interpolation ──

    /**
     * DARE fanout interpolation.
     *
     * @param fanOuts      float[256] non-uniform fanout table
     * @param positionNorm normalized position in [0, 1]
     * @return fanout for the node at this position
     */
    public static int dareFanout(float[] fanOuts, double positionNorm) {
        double pred = positionNorm * (fanOuts.length - 1);   // [0, FANOUT_TABLE_SIZE-1]
        int left = Math.max(0, Math.min(fanOuts.length - 1, (int) pred));
        int right = Math.min(fanOuts.length - 1, left + 1);
        double frac = pred - left;
        double fanout = fanOuts[left] * (1.0 - frac) + fanOuts[right] * frac;
        return Math.max(1, (int) Math.round(fanout));
    }

    // ── Temperature Labeling ──

    /**
     * Two-pass temperature labeling for all regions.
     * First pass: compute frequency per region from PDF density.
     * Second pass: compute T_ij and set isHot.
     *
     * @param regions    per-bucket regions with lo/hi set; freq and isHot are filled
     * @param globalPdf  float[16384] data distribution
     * @param dataSize   total records
     * @param kMin       global key range lower bound
     * @param kMax       global key range upper bound
     */
    public static void labelTemperatures(Region[] regions, float[] globalPdf,
                                          float dataSize, double kMin, double kMax) {
        int pdfLen = globalPdf.length;
        double range = kMax - kMin;

        // Pass 1 — compute frequency for each region
        double freqMax = 0;
        for (Region r : regions) {
            int loIdx = Math.max(0, (int) (pdfLen * (r.lo - kMin) / range));
            int hiIdx = Math.min(pdfLen - 1, (int) (pdfLen * (r.hi - kMin) / range));
            double sum = 0;
            for (int i = loIdx; i <= hiIdx; i++) {
                sum += globalPdf[i];
            }
            r.freq = sum * dataSize;  // expected op count in this region
            if (r.freq > freqMax) freqMax = r.freq;
        }

        // Pass 2 — label
        if (freqMax == 0) freqMax = 1;  // avoid div-by-zero
        for (Region r : regions) {
            double A = Math.log(1.0 + r.freq) / Math.log(1.0 + freqMax);
            double R = (r.hi - r.lo) / range;
            double T = A * (1.0 + BETA * R);
            r.isHot = T >= TAU;
        }
    }
}
