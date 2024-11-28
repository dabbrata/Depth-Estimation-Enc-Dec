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
1. <b><a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/1_data-preprocessing.ipynb">Data preprocessing</a></b> : 
All images have their bounding boxes in the PASCAL VOC format and their info saved in `XML` format in `annotaions` directory. 
Only the region bounded by bounding box taken as input and their respective labels taken as output.

2. <b><a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/2_training-face-mask-model.ipynb">Training mask detector model</a></b> :
Transfer learning was used to train the inputs. The classifier model was built with <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3">InceptionV3</a> neural network architecture.
After training for 20 epochs, accuracy on test set was 96.81%.

3. <b><a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/3.2_detecting-mask-w-mtcnn.ipynb">Detecting face mask </a><a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/3.1_detecting-mask-w-haarcascade.ipynb">in real-time</a></b> :
First task was to detect faces from each frame of the video. 
At first I used <a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/3.1_detecting-mask-w-haarcascade.ipynb">Haarcascade classifer</a> from OpenCV for face detection. Average FPS I got while running on my machine was around 16. 
But face detection wasn't that accurate. This classifer struggled detecting faces with mask. In low-light condition it struggled the most.
<br/>Then I tried <a href="https://github.com/FarhanSadaf/face-mask-detection/blob/master/3.2_detecting-mask-w-mtcnn.ipynb">MTCNN</a> for face detection. 
This algorithm performed great detecting faces, even in the low light. But while running on my machine, the average FPS I got was about 1.4. 
Which is pretty slow comparing with haarcascade classifier. 

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
