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
- UBNormal dataset

Dataset structure:

    Data/
    └── Avenue/
        ├── training_frames/       
  
        └── testing_frames/


## Training

```bash
python train.py
