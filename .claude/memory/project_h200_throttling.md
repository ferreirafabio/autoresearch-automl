---
name: H200 power throttling causes lower baseline val_bpb
description: H200 SM clocks throttle to ~1600MHz (vs H100 ~1950MHz) due to HBM3e power draw, causing 18% fewer training steps and ~0.01 higher baseline val_bpb
type: project
---

H200 GPUs on kislurm (dlc2gpu18/19/20) power-throttle SM clocks from 1980 MHz max to ~1575-1635 MHz under sustained load. Both H200 and H100 have 700W TDP and the same GH100 compute die, but H200's HBM3e (4.8 TB/s, 141GB) draws more power than H100's HBM3 (3.35 TB/s, 80GB), leaving less power for SMs.

**Why:** Investigated because our baseline val_bpb=1.008 vs Karpathy/Ravid's 0.998 with identical config and code.

**How to apply:** When comparing results against H100 benchmarks, note the ~18% throughput penalty. Clock-adjusted MFU is 40.3% (≈ Karpathy's 39.8%). The baseline offset doesn't affect HPO convergence quality — our TPE still reaches 0.9775.

Key numbers:
- Our H200: ~1600 MHz, 782 steps/300s, MFU=32.6% (40.3% clock-adjusted)
- Karpathy H100: ~1950 MHz, 953 steps/300s, MFU=39.8%
- Throughput ratio: 1.22x (H100 faster due to higher sustained clocks)
