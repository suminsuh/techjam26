# Error analysis note

Operating threshold (frozen from clean val): `0.0247`

## False positives (authentic flagged as AIGC)

These are the highest-confidence accusations. Typical causes: heavy
compression already present in the 'real' file, overly smooth texture,
or screenshots that look like generated images.

- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img163784jpg_1317.jpg`  pred=0.954  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img160573jpg_738.jpg`  pred=0.628  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img160385jpg_129.jpg`  pred=0.579  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img161574jpg_217.jpg`  pred=0.562  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img161934jpg_996.jpg`  pred=0.362  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img162477jpg_161.jpg`  pred=0.342  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img162272jpg_1010.jpg`  pred=0.268  label=real
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/REAL/coco_val2017img162701jpg_686.jpg`  pred=0.259  label=real

## False negatives (AIGC missed)

These are the lowest-confidence misses. Typical causes: JPEG 30–50,
strong blur, or generators whose artifacts look like camera noise.
After redistribution, the model often becomes conservative (predicts real).

- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advancedfdf4bcadfd808ac1eb64836b80a0cdf6jpg_605.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advancedfdf4bcadfd808ac1eb64836b80a0cdf6jpg_736.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advanced949834d1ceb497d32c313c031419a332jpg_506.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advanced7524213fb3b385a99e07bb92c32326abjpg_160.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advancedd5c4406922e8a1a70608e3a38a06e939jpg_540.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advanced54be30e6868c0ebc289beba6f04dc2fdjpg_412.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advanced1ceee4a4076b1977e262547376a0449ajpg_718.jpg`  pred=0.000  label=aigc
- `C:/Users/HL/Desktop/project/techjam26/data/wildfake_holdout/FAKE/dalle3_advanceda3c8838644be5316a5e9e0c178252fe2jpg_751.jpg`  pred=0.000  label=aigc

## Trade-offs to discuss in the write-up

- Forensic cues win on clean images and lose under JPEG / blur.
- The spatial stream is more stable but weaker on unseen generators.
- A low-FPR threshold (default 5%) protects creators and will miss
  some fakes. Raise it only if the product priority flips.
- Do not retune the threshold per transform when quoting robustness.

