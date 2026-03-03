#!/usr/bin/env python3
"""
Generate /tmp/ane_sram_{ch}ch_{sp}sp.mlpackage models for ANE benchmarks.

Each model is a single 1x1 conv: fp32_in -> cast_fp16 -> conv -> cast_fp32 -> out
Covers all configs needed by inmem_basic, inmem_bench, sram_bench, sram_probe.
"""

import numpy as np
import os
import sys

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
except ImportError:
    print("ERROR: coremltools not installed. Install with: pip install coremltools", file=sys.stderr)
    sys.exit(1)

CONFIGS = [
    (256, 64), (512, 64), (1024, 64), (1536, 64),
    (2048, 64), (2560, 64), (3072, 64), (3584, 64),
    (4096, 64), (4608, 64), (5120, 64), (6144, 64),
    (8192, 32),
]


def gen_model(ch, sp):
    """Build a coremltools MIL model with a single 1x1 conv."""

    @mb.program(
        input_specs=[mb.TensorSpec(shape=(1, ch, 1, sp), dtype=types.fp32)],
        opset_version=ct.target.iOS18,
    )
    def prog(x):
        x_fp16 = mb.cast(x=x, dtype="fp16", name="cast_in")
        w = np.random.randn(ch, ch, 1, 1).astype(np.float16) * 0.01
        c = mb.conv(
            x=x_fp16,
            weight=w,
            pad_type="valid",
            strides=[1, 1],
            dilations=[1, 1],
            groups=1,
            name="c0",
        )
        out = mb.cast(x=c, dtype="fp32", name="cast_out")
        return out

    model = ct.convert(
        prog,
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )
    return model


def main():
    created = 0
    skipped = 0

    print(f"Generating {len(CONFIGS)} mlpackage models in /tmp/...")

    for ch, sp in CONFIGS:
        path = f"/tmp/ane_sram_{ch}ch_{sp}sp.mlpackage"
        w_mb = ch * ch * 2 / 1024 / 1024

        if os.path.exists(path):
            print(f"  [skip] {ch}ch x {sp}sp (exists)")
            skipped += 1
            continue

        print(f"  [gen]  {ch}ch x {sp}sp  (weights: {w_mb:.1f} MB)...", end="", flush=True)
        try:
            model = gen_model(ch, sp)
            model.save(path)
            print(" OK")
            created += 1
        except Exception as e:
            print(f" FAILED: {e}")

    print(f"\nDone: {created} created, {skipped} skipped (already existed).")
    return 0 if created + skipped == len(CONFIGS) else 1


if __name__ == "__main__":
    sys.exit(main())
