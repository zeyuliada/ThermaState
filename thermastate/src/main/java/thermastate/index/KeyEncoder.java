/*
 * Copyright (c) 2025-2026 ADA Lab, Soochow University
 */

package thermastate.index;

/**
 * Encodes Flink state access key as a double for the learned index.
 *
 * Packs the paper's {@code ⟨kg, sid, uk⟩} triple into 52 bits so every
 * integer is exactly representable in IEEE 754 double (mantissa = 53 bits;
 * using 52 leaves a 1-ULP safety margin).
 *
 * <pre>
 * Bit layout (LSB first):
 * ┌──────────────┬─────────────┬───────────────────┐
 * │  kg (12 bit) │ sid (8 bit) │   uk (32 bit)      │
 * │   [51:40]    │  [39:32]    │    [31:0]          │
 * └──────────────┴─────────────┴───────────────────┘
 *
 * kg  — Flink key group index   [0, 4095]
 * sid — state descriptor id     [0, 255]
 * uk  — user-key hash           [0, 2³² − 1]
 * </pre>
 *
 * Because value = decode(encode(kg, sid, uk)) is an exact round-trip,
 * the index operates on raw doubles without hash collisions.
 */
public final class KeyEncoder {

    // ── Bit allocation ──
    public static final int KG_BITS  = 12;
    public static final int SID_BITS = 8;
    public static final int UK_BITS  = 32;           // 12 + 8 + 32 = 52

    // ── Max values ──
    public static final int MAX_KEY_GROUP      = (1 << KG_BITS)  - 1;  // 4095
    public static final int MAX_STATE_ID       = (1 << SID_BITS) - 1;  // 255
    public static final long MAX_USER_KEY      = (1L << UK_BITS) - 1;  // 4,294,967,295

    // ── Shift offsets ──
    private static final int UK_SHIFT  = 0;
    private static final int SID_SHIFT = UK_BITS;             // 32
    private static final int KG_SHIFT  = SID_BITS + UK_BITS;  // 40

    // ── Masks ──
    private static final long UK_MASK  = ((1L << UK_BITS)  - 1) << UK_SHIFT;
    private static final long SID_MASK = ((1L << SID_BITS) - 1) << SID_SHIFT;
    private static final long KG_MASK  = ((1L << KG_BITS)  - 1) << KG_SHIFT;

    private KeyEncoder() {}

    // ── Encode ──

    /** Pack {@code ⟨kg, sid, uk⟩} → exact double. */
    public static double encode(int kg, int sid, long uk) {
        long bits = ((long)(kg & MAX_KEY_GROUP) << KG_SHIFT)
                  | ((long)(sid & MAX_STATE_ID)  << SID_SHIFT)
                  | (uk & MAX_USER_KEY);
        return (double) bits;
    }

    /** Pack for single-machine use (kg = 0). */
    public static double encode(int sid, long uk) {
        return encode(0, sid, uk);
    }

    /** Pack user-key hash directly without sid (backward-compatible). */
    public static double encode(long uk) {
        return encode(0, 0, uk);
    }

    // ── Decode ──

    /** Extract key group from encoded key. */
    public static int keyGroup(double key) {
        return (int)(((long) key & KG_MASK) >>> KG_SHIFT);
    }

    /** Extract state descriptor id from encoded key. */
    public static int stateId(double key) {
        return (int)(((long) key & SID_MASK) >>> SID_SHIFT);
    }

    /** Extract user-key hash from encoded key. */
    public static long userKey(double key) {
        return (long) key & UK_MASK;
    }

    // ── Component representation ──

    /** Immutable decoded triple. */
    public static final class Decoded {
        public final int kg;
        public final int sid;
        public final long uk;

        Decoded(int kg, int sid, long uk) {
            this.kg = kg;
            this.sid = sid;
            this.uk = uk;
        }

        /** Decode a previously encoded double. */
        public static Decoded from(double key) {
            long bits = (long) key;
            return new Decoded(
                (int)((bits & KG_MASK) >>> KG_SHIFT),
                (int)((bits & SID_MASK) >>> SID_SHIFT),
                bits & UK_MASK
            );
        }

        @Override
        public boolean equals(Object o) {
            if (o instanceof Decoded) {
                Decoded d = (Decoded) o;
                return kg == d.kg && sid == d.sid && uk == d.uk;
            }
            return false;
        }

        @Override
        public int hashCode() {
            return (kg * 31 + sid) * 31 + Long.hashCode(uk);
        }

        @Override
        public String toString() {
            return String.format("⟨kg=%d,sid=%d,uk=%d⟩", kg, sid, uk);
        }
    }
}
