# 3D Point Cloud Semantic Segmentation using SparseConvolution based Encoder-Decoder
The source code for 3D point cloud semantic segmentation using SparseConv, required for GSOC qualification assignment.


**Highlights of the implementation:**

* Integrated with [Open3D-ML](http://www.open3d.org/docs/release/open3d_ml.html).
* Closely follows the pipeline of models implemented in Open3D-ML.
* Submanifold Sparse Convolution is adopted for better performance.
* Contains code for 3D visualization of the prediction.

## Results

The task was to overfit a single point cloud sample. The provided model ran for **300 epochs** and the final reported accuracy of overfitted model is **0.79**.

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
