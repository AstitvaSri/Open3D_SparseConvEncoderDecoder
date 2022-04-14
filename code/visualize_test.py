import open3d as o3d
import numpy as np
import cv2
import os

from torch import gt

data = np.load('../data/train/data.npy')

xyz = data[:,:3]
rgb = data[:,3:6]/255
gt_labels = data[:,6]

# visualize colored point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
# o3d.visualization.draw_geometries([pcd])

labels = np.load('predictions/predicted_labels.npy')

eq = gt_labels == labels
print("Prediction Accuracy = ", eq.sum()/len(eq))


# visualize labeled point cloud with fake colors
vis_colors = np.array([[255,0,0], [0,255,0], [0,0,255], [255,255,0], [0,255,255], [255,0,255], [255,127,255], [127,255,255], [127,127,255], [127,255,127], [255,127,127], [127,127,127], [127,127,255]])
labels_color = np.zeros((labels.shape[0],3))
for i in range(len(vis_colors)):
    labels_color[labels==i] = vis_colors[i]/255

pcd.colors = o3d.utility.Vector3dVector(labels_color)
o3d.visualization.draw_geometries([pcd])