# Improvd RHCNet with memory augummented E3D-LSTM for prediction based Video Anomaly Detection

PyTorch implementation of a hybrid spatiotemporal autoencoder for video anomaly detection.

The architecture combines:

- Improved Residual Hybrid Convolution Network (RHCNet)
- Memory Augumented Eidetic 3D-LSTM based spatiotemporal modeling

The model predicts the next frame given previous frames and detects anomalies using reconstruction error.

## Architecture

Encoder:
Improved RHCNet-based hierarchical feature extraction.

Temporal modeling:
Memory Augumented E3D-LSTM.

Decoder:
Improved RHCNet blocks.


## Dataset

Experiments were conducted on:

- CUHK Avenue dataset
- ShanghiTech dataset
- Ubnormal dataset

Dataset structure:

    Data/
    └── UCSD/
        ├── training_frames/
            └── Train001/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train002/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train004/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
            └── Train005/
                └── 001.tif/
                └── 002.tif/
                └── 003.tif/
                └── 004.tif/
        └── validation_frames/
                └── Train002/
                    └── 001.tif/
                    └── 002.tif/
                    └── 003.tif/
                    └── 004.tif/
        └── testing_frames/
                └── Test001/
                    └── 001.tif/
                    └── 002.tif/
                    └── 003.tif/
                    └── 004.tif/

## Training

```bash
python train.py
