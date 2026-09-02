/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.reorganizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import thermastate.index.Index;

/**
 * Global tree rebuild — triggered when DriftDetector signals g_t ≥ τ_g for P
 * consecutive checks.
 *
 * Snapshot all (key, bundleId) pairs, re-sort, bulkLoad to rebuild tree.
 * The ValueBundleBlock is NOT touched — bundle IDs remain valid.
 */
public final class GlobalRebuilder {

    private static final Logger LOG = LoggerFactory.getLogger(GlobalRebuilder.class);

    private GlobalRebuilder() {}

    /** Rebuild the entire index tree. Returns true if rebuild succeeded. */
    public static boolean rebuild(Index index) {
        List<Index.IntEntry> entries = new ArrayList<>();
        index.forEach((key, bundleId) -> {
            entries.add(new Index.IntEntry(key, bundleId));
        });

        if (entries.isEmpty()) {
            LOG.info("Global rebuild skipped: tree is empty");
            return false;
        }

        entries.sort(Comparator.comparingDouble(e -> e.key));
        LOG.info("Global rebuild: {} entries snapshot", entries.size());

        try {
            index.bulkLoad(entries);
        } catch (Exception e) {
            LOG.error("Global rebuild failed: {}", e.getMessage());
            return false;
        }

        LOG.info("Global rebuild complete: {} entries, new root fanout={}",
                 entries.size(), index.root().capacity());
        return true;
    }
}
