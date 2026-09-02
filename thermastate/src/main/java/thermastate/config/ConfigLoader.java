/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.config;

import thermastate.index.Configuration;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Loads a C++-exported JSON Configuration for ThermaStateBackend.
 *
 * JSON format:
 * {
 *   "rootFanOut": 64.0,
 *   "innerFanouts": [8.0, 8.0, ...]
 * }
 */
public class ConfigLoader {

    public static Configuration load(String path) throws IOException {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = r.readLine()) != null) {
                sb.append(line.trim());
            }
        }
        return parse(sb.toString());
    }

    static Configuration parse(String json) {
        Configuration conf = new Configuration();

        // rootFanOut
        int rfIdx = json.indexOf("\"rootFanOut\"");
        if (rfIdx >= 0) {
            int colon = json.indexOf(":", rfIdx);
            int end = json.indexOf(",", colon);
            if (end < 0) end = json.indexOf("}", colon);
            String val = json.substring(colon + 1, end).trim();
            conf.rootFanOut = Float.parseFloat(val);
        }

        // innerFanouts array — per-root-bucket TS-MDP decisions
        int arrIdx = json.indexOf("\"innerFanouts\"");
        if (arrIdx >= 0) {
            int bracket = json.indexOf("[", arrIdx);
            int close = json.indexOf("]", bracket);
            String arr = json.substring(bracket + 1, close);
            String[] parts = arr.split(",");
            conf.innerFanouts = new int[parts.length];
            for (int i = 0; i < parts.length; i++) {
                String t = parts[i].trim();
                if (!t.isEmpty()) conf.innerFanouts[i] = Integer.parseInt(t);
            }
        }

        return conf;
    }
}
