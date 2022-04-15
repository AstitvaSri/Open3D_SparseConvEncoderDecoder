# 3D Point Cloud Semantic Segmentation using SparseConvolution based Encoder-Decoder
The source code for 3D point cloud semantic segmentation using SparseConv, required for GSOC qualification assignment.


**Highlights of the implementation:**

* Integrated with [Open3D-ML](http://www.open3d.org/docs/release/open3d_ml.html).
* Closely follows the pipeline of models implemented in Open3D-ML.
* Submanifold Sparse Convolution is adopted for faster convergence.
* GPU support.
* Contains code for 3D visualization of the prediction.

## Results

The task was to overfit a single point cloud sample. The implemented model ran for **300 epochs** and following are the final reported metrices:\
**Training Loss = 0.601**\
**Validation Loss = 0.600**\
**Mean accuracy:** Train = 0.757, Validation = 0.758\
**Mean IOU:** Train = 0.575, Validation = 0.575\
**Test accuracy of the overfitted model = 0.793**

![Alt Text](https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/pcd.gif)

## Insights
* Submanifold SparseConv converges faster as it convolves only when kernel's center is at the active site.
* Weighted-cross-entropy loss works better and provides easier convergence as the given sample point cloud has **class-imbalance** problem. When all the classes were weighted equally, the model seemed to be stuck in a local minima with mean accuracy as low as **0.1**. First, I tried [sklearn's compute_class_weight](https://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html) function, but it didn't help the model to come out of local minima. Finally, I used **number of points per class as the class-weights** and the model started converging.
* Using a larger learning rate also seemed to be one of the reasons leading to poor convergence as the model just oscillates back and forth. Smaller learning rate with exponential decay proved useful.

## Plots
### Training & Validation Loss
<kbd>
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/train_loss.png">
</kbd>
<kbd>
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_loss.png">()
</kbd>

### Training & Validation Accuracy
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/training_acc.png">
</kbd>
<kbd>
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_acc.png">()
</kbd>

### Training & Validation IoU
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/training_IOU.png">
</kbd>
<kbd>
<img src="https://github.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_IOU.png">()
</kbd>.com/AstitvaSri/Open3D_SparseConvEncoderDecoder/blob/main/plots/val_IOU.png)

## Setup

```bash
# clone this repository
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
