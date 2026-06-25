# `FHKMNet`

PyTorch implementation of `FHKMNet` for referring camouflaged object detection.

## Framework
The framework figure and qualitative results will be added to the repository together with the final project assets.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation
- Download the Ref-COD dataset from [zhangxuying1004/RefCOD](https://github.com/zhangxuying1004/RefCOD).
- Place the dataset under `./dataset/R2C7K`.
- Place the PVTv2-B2 pretrained weight at `./pvt_weights/pvt_v2_b2.pth`.

## Training
Run:

```bash
python train_fhkm.py
```

## Testing
Run:

```bash
python test.py
```

## Inference
Run:

```bash
python infer.py
```

## Notes
- The dataset is not included in this repository.
- The pretrained backbone weight is not included in this repository and should be uploaded separately later.
