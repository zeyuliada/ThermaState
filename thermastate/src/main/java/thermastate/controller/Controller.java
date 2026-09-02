/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.controller;

import com.sun.jna.Pointer;
import thermastate.agent.ThermaStateInference;
import thermastate.index.Configuration;

/**
 * Genetic algorithm controller — delegates to native C++ GA for optimal
 * Configuration search via the already-loaded Q-Network.
 *
 * Usage:
 *   Pointer handle = ThermaStateInference.INSTANCE.thermastate_load(modelPath, scalarPath);
 *   Controller ctrl = new Controller(ThermaStateInference.INSTANCE, handle);
 *   Configuration best = ctrl.search(pdf, dataSize);
 */
public class Controller {

    private final ThermaStateInference lib;
    private final Pointer handle;

    public Controller(ThermaStateInference lib, Pointer handle) {
        this.lib = lib;
        this.handle = handle;
    }

    /**
     * Run native GA to find the best Configuration for the given data distribution.
     *
     * @param distribution  float[16384] PDF histogram of the dataset key distribution
     * @param dataSize      total number of key-value records
     * @return best Configuration found by GA, or default on failure
     */
    public Configuration search(float[] distribution, float dataSize) {
        float[] outRoot = new float[1];
        float[] outInner = new float[Configuration.INNER_FANOUT_SIZE];

        int rc = lib.thermastate_search(handle, distribution, dataSize, outRoot, outInner);
        if (rc != 0) {
            return Configuration.defaultConfiguration();
        }

        Configuration conf = new Configuration();
        conf.rootFanOut = outRoot[0];
        for (int c = 0; c < Configuration.INNER_FANOUT_COLUMN; c++) {
            conf.fanOuts[0][c] = outInner[c];
        }
        return conf;
    }
}
