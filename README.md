# 3D Point Cloud Semantic Segmentation using SparseConv Encoder-Decoder
The source code for 3D point cloud semantic segmentation using SparseConv, required for GSOC qualification assignment.


**Highlights of the implementation:**

* Integrated with Open3D-ML.
* Closely follows pipeline format use to run other models in Open3D-ML.
* Submanifold Sparse Convolution is utilised.
* Contains code for 3D visualization of the prediction.


## Setup

```bash
# clone the repository
git clone https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder.git

# environment setup
cd Open3D_SparseConvEncoderDecoder
conda create -f environment.yml
```
## Training
```bash
python train.py --data_path ../data --ckpt_path ./logs/SparseEncDec_Semantic3D_torch/checkpoint
```
## Testing
```bash
python test.py --data_path ../data --ckpt_path ./logs/SparseEncDec_Semantic3D_torch/checkpoint
```
## Visualizing Prediction
```bash
python visualize_test.py
```
