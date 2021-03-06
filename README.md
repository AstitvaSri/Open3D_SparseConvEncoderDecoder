# 3D Point Cloud Segmentation using Sparse Convolution.


**Highlights of the implementation:**

* Integrated with [Open3D-ML](http://www.open3d.org/docs/release/open3d_ml.html).
* Closely follows the pipeline of models implemented in Open3D-ML.
* GPU support.
* Contains code for 3D visualization of the prediction.

## Results

The provided model is overfitted a single point cloud sample, which ran for **300 epochs** and following are the final reported metrices:\
**Final Training Loss:** 0.601\
**Final Validation Loss:** 0.600\
**Final Mean accuracy:** Train = 0.757, Validation = 0.758\
**Final Mean IOU:** Train = 0.575, Validation = 0.575\
**Test accuracy of the overfitted model = 0.793**

<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/pcd.gif" width="1920">
</kbd>

## Insights
* Submanifold SparseConv feed-forwards faster as it convolves only when kernel's center is at the active site.
* Weighted-cross-entropy loss works better and provides easier convergence as the given sample point cloud has **class-imbalance** problem. When all the classes were weighted equally, the model seemed to be stuck in a local minima with mean accuracy as low as **0.1**. First, I tried [sklearn's compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) function, but it didn't help the model to come out of the local minima. Finally, I used **number of points per class as the class-weights** and the model started converging.
* Using a larger learning rate also seemed to be one of the reasons leading to poor convergence, as the model just oscillates back and forth. Smaller learning rate with exponential decay proved useful.

## Plots
### Training & Validation Loss
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/train_loss.png">
</kbd>
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_loss.png">
</kbd>

### Training & Validation Accuracy
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/training_acc.png">
</kbd>
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_acc.png">
</kbd>

### Training & Validation IoU
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/training_IOU.png">
</kbd>
<kbd>
  <img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_IOU.png">
</kbd>

## Setup

```bash
cd $HOME

# clone this repository
git clone https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder.git

# anaconda environment
cd $HOME/Open3D_SparseConvEncoderDecoder
conda create -f environment.yml
conda activate open3dml

# clone Open3D-ML
git clone https://github.com/isl-org/Open3D-ML.git
export OPEN3D_ML_ROOT=$HOME/Open3D-ML
```

## Training
```bash
cd $HOME/Open3D_SparseConvEncoderDecoder/code
python train.py --data_path ../data --ckpt_path ./logs/SparseEncDec_Semantic3D_torch/checkpoint
```
## Testing
```bash
cd $HOME/Open3D_SparseConvEncoderDecoder/code
python test.py --data_path ../data --ckpt_path ./logs/SparseEncDec_Semantic3D_torch/checkpoint
```
## Visualizing Prediction
```bash
cd $HOME/Open3D_SparseConvEncoderDecoder/code
python visualize_test.py
```
