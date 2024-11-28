# Depth Estimation from Monocular Images with Enhanced Encoder-Decoder Architecture

## Overview
Monocular Depth Estimation refers to the process of predicting depth information from a single 2D image. This technique has wide applications in areas like robotics, autonomous driving, and augmented reality. It allows machines to understand the 3D structure of a scene without requiring expensive stereo cameras or depth sensors.
In this simple project, an input image is processed by a deep learning model to estimate the depth map, which encodes the relative distances of objects in the scene.

## Quickstart the project
1. Download the code in `ZIP` or open with `GitHub Desktop` or `https://github.com/dabbrata/Depth-Estimation-Enc-Dec.git`.
2. Then import `IRv2_Decoder_Monocular_Depth_Estimation_Full_Notebook.ipynb` file to your notebook.
3. Install required python dependencies into your python environment / virtual environment using `pip install -r Requirements.txt`.
4. Run all the cells of that imported (.ipynb) file.

## Dataset
The dataset used to train the depth estimation model taken from <a href="https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2">here</a>.


This dataset contains 1449 densely labeled pairs of aligned RGB and depth images, 464 new scenes taken from 3 cities, 
407,024 new unlabeled frames and Each object is labeled with a class and an instance number (cup1, cup2, cup3, etc).

## Workflow
1. <b>Data preprocessing</b> :
a. Augmentation: Horizontal flipping adds variety to the training dataset and helps the model generalize better.
b. Normalization Consistency: Both RGB images and depth maps are normalized to the same scale ([0, 1]), ensuring compatibility during training.
c. Resizing: Ensures uniform dimensions for both images and depth maps, critical for batch processing in deep learning models.
d. Grayscale Conversion: Reduces depth maps to a single channel, reflecting their intrinsic nature as 2D scalar fields.

2. <b>Training depth estimation model</b> :
Transfer learning was used to train the inputs. Here, the pretrained model `Inception Resnet v2` is used as encoder for feature extraction.
After training for 15 epochs, the accuracy for three kinds of thresholds are:
a. Delta1: 89.3%
b. Delta2: 96.7%
c. Delta3: 98.5%

3. Initially, I used a basic convolutional encoder-decoder architecture for this task. While the model was simple and lightweight, the predicted depth maps lacked accuracy, especially in regions with fine details or complex textures.
To improve the accuracy, I implemented an enhanced encoder-decoder architecture with attention mechanisms and skip connections. This model significantly improved depth prediction quality, especially in areas with intricate details and varying lighting conditions. However, the computational complexity increased.
Future improvements may include optimizing the model with more fine tunning.
## Results
<table>
<tr>
<th>Input</th>
<th>Output</th>
</tr>
<tr>
<td><img src="images/test.jpg"/></td>
<td><img src="images/test-result.png"/></td>
</tr>
</table>

## Links and References
- Face Mask Detection dataset: https://www.kaggle.com/andrewmvd/face-mask-detection
- InceptionV3: https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3
- Face Detection using Haar Cascades: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
- How to Perform Face Detection with Deep Learning: https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
- GitHub MTCNN: https://github.com/ipazc/mtcnn

## Licensing
The code in this project is licensed under [MIT License](LICENSE).
