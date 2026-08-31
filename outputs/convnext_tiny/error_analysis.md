# Error analysis note

Operating threshold (frozen from clean val): `0.0852`

## False positives (authentic flagged as AIGC)

These are the highest-confidence accusations. Typical causes: heavy
compression already present in the 'real' file, overly smooth texture,
or screenshots that look like generated images.

- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_1_ef1238b1f320fa40.jpg`  pred=0.952  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_1_5fc2cc3aca4c56c4.jpg`  pred=0.795  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_0_79fd61c8155b1849.jpg`  pred=0.758  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_0_7f5f3ddab1a574ca.jpg`  pred=0.641  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_1_2e396230de328ecf.jpg`  pred=0.600  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_0_2c2542c44b0c4ea4.jpg`  pred=0.517  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_0_e1e0c52b009efb9b.jpg`  pred=0.506  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/REAL/val_1_a12bc2c295b94e4c.jpg`  pred=0.481  label=real

## False negatives (AIGC missed)

These are the lowest-confidence misses. Typical causes: JPEG 30–50,
strong blur, or generators whose artifacts look like camera noise.
After redistribution, the model often becomes conservative (predicts real).

- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/FAKE/val_1_full_synthetic_009230.jpg`  pred=0.001  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/FAKE/val_0_full_synthetic_009567.jpg`  pred=0.002  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/sid_set/val/FAKE/val_0_full_synthetic_007794.jpg`  pred=0.015  label=aigc

## Trade-offs to discuss in the write-up

- Forensic cues win on clean images and lose under JPEG / blur.
- The spatial stream is more stable but weaker on unseen generators.
- A low-FPR threshold (default 5%) protects creators and will miss
  some fakes. Raise it only if the product priority flips.
- Do not retune the threshold per transform when quoting robustness.

