# Enhanced Encoder-Decoder Architecture for Accurate Monocular Depth Estimation 

## Overview
Monocular Depth Estimation predicts depth information from a single 2D image, providing 3D scene understanding without the need for stereo cameras or depth sensors. This method is useful in robotics, autonomous driving, and augmented reality, where spatial awareness is essential. In this project, we use a deep learning-based encoder-decoder architecture for depth map generation. The encoder, based on the pretrained Inception-ResNet-v2 (IRv2), extracts multi-scale features from the input image, capturing both local and global image patterns. The decoder processes these features to reconstruct a high-resolution depth map, preserving both scene context and fine details. The model is optimized for efficient inference, making it suitable for real-time applications.

#### Enc-Dec-IRv2 Architecture:
<img src="Images/net_architecture.png" /><br/>
#### Schematic Diagram of IRv2:
<img src="Images/algo_block2.png" />

## Quickstart the project
1. Download the code in `ZIP` or open with `GitHub Desktop` or `https://github.com/dabbrata/Depth-Estimation-Enc-Dec.git`.
2. Then import `IRv2_Decoder_Monocular_Depth_Estimation_Full_Notebook.ipynb` file to your notebook.
3. Install required python dependencies into your python environment / virtual environment using `pip install -r Requirements.txt`.
4. Run all the cells of that imported (.ipynb) file.

## Dataset
The dataset used to train the depth estimation model: 

#### NYU Depth V2 <a href="https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2">Link</a>.
This dataset contains 1449 densely labeled pairs of aligned RGB and depth images, 464 new scenes taken from 3 cities, 
407,024 new unlabeled frames and Each object is labeled with a class and an instance number (cup1, cup2, cup3, etc).

#### KITTI <a href="https://www.kaggle.com/datasets/garymk/kitti-3d-object-detection-dataset">Link1</a>, <a href="https://www.kaggle.com/datasets/emadmous/kitti-sfd-train">Link2</a>.
For the application of our model on the KITTI dataset, we selected a subset of images from the entire dataset. Specifically, we utilized 7,281 images for training and an additional 200 images for testing the model's performance.

#### Cityscapes <a href="https://www.kaggle.com/datasets/andreabaccolo/cityscapes">Link</a>.
In applying our model to the Cityscapes dataset, we selected a portion of the dataset for evaluation. Specifically, 1,571 images were used for training, and 500 images were reserved for testing the model’s performance.

## Workflow
<body>
    <h4><b>1. Data preprocessing: </b></h4>
    <ul>
        <li>a. Augmentation: Horizontal flipping adds variety to the training dataset and helps the model generalize better.</li>
        <li>b. Normalization Consistency: Both RGB images and depth maps are normalized to the same scale ([0, 1]), ensuring compatibility during training.</li>
        <li>c. Resizing: Ensures uniform dimensions for both images and depth maps, critical for batch processing in deep learning models.</li>
        <li>d. Grayscale Conversion: Reduces depth maps to a single channel, reflecting their intrinsic nature as 2D scalar fields.</li>
    </ul>
</body>




 
<body>
    <h4><b>2. Training depth estimation model: </b></h4>
    <p>Transfer learning was used to train the inputs. The pretrained model <b>Inception ResNet v2</b> was used as the encoder for feature extraction on the NYU-Depth v2 dataset. After training for 15 epochs,         the accuracy for three kinds of thresholds are as follows:</p>
    <ul>
        <li><b>Delta1:</b> 89.3%</li>
        <li><b>Delta2:</b> 96.7%</li>
        <li><b>Delta3:</b> 98.5%</li>
    </ul>
    <p> The proposed model is trained on both the KITTI and Cityscapes outdoor datasets. On the KITTI dataset, our model was compared with the DINOv2 transformer model and demonstrated superior efficiency while          maintaining competitive accuracy. Specifically, the IRv2 model achieves an inference time of 0.019 seconds, making it 23 times faster than the DINOv2-S transformer model for the same input dimensions.            Although our model exhibits slightly lower accuracy compared to DINOv2-S, the trade-off in speed significantly enhances its practicality. The accuracy of DINOv2-S is as follows: </p>
    <ul>
        <li><b>Delta1:</b> 89.9%</li>
        <li><b>Delta2:</b> 97.2%</li>
        <li><b>Delta3:</b> 98.9%</li>
    </ul>
</body>

<body>
<!--     <h4><b>2. Training depth estimation model: </b></h4> -->
    <p>3. Initially, we used a basic convolutional encoder-decoder architecture for this task. While the model was simple and lightweight, the predicted depth maps lacked accuracy, especially in regions with         fine details or complex textures. To improve the accuracy, we implemented an enhanced encoder-decoder architecture with attention mechanisms and skip connections. This model significantly improved depth          prediction quality, especially in areas with intricate details and varying lighting conditions. However, the computational complexity increased. Future improvements may include optimizing the model with more     fine tunning.</p>
</body>

## Results
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
<!--     <title>Enc-Dec-IRv2 on NYU-Depth V2</title> -->
</head>
<body>
    <table>
        <tr>
            <td>
                <h4>Enc-Dec-IRv2 on NYU-Depth V2</h4>
                <table>
                    <tr>
                        <th>Input</th>
                        <th>Output</th>
                    </tr>
                    <tr>
                        <td><img src="Images/rgb.png" alt="Input Image"/></td>
                        <td><img src="Images/high.png" alt="Output Image"/></td>
                    </tr>
                </table>
            </td>
            <td>
                <h4>Enc-Dec-IRv2 on KITTI</h4>
                <table>
                    <tr>
                        <th>Input</th>
                        <th>Output</th>
                    </tr>
                    <tr>
                        <td><img src="Images/k3_rgb.png" alt="Input Image"/></td>
                        <td><img src="Images/k3_out.png" alt="Output Image"/></td>
                    </tr>
                </table>
            </td>
            <td>
                <h4>Enc-Dec-IRv2 on Cityscapes</h4>
                <table>
                    <tr>
                        <th>Input</th>
                        <th>Output</th>
                    </tr>
                    <tr>
                        <td><img src="Images/city_rgb.png" alt="Input Image"/></td>
                        <td><img src="Images/city_out.png" alt="Output Image"/></td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>

https://github.com/user-attachments/assets/fb87f76a-f076-4ece-917b-eb40cfc5083d



## Links and References
- Depth estimation dataset (NYU Depth v2): https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2
- Depth estimation dataset (KITTI): https://www.kaggle.com/datasets/garymk/kitti-3d-object-detection-dataset, https://www.kaggle.com/datasets/emadmous/kitti-sfd-train
- Depth estimation dataset (Cityscapes): https://www.kaggle.com/datasets/andreabaccolo/cityscapes
- Inception-Resnet-v2: https://paperswithcode.com/method/inception-resnet-v2
- Relevant Paper: https://ieeexplore.ieee.org/document/9376365

## Contributors
1. [Argho Deb Das](https://github.com/MrArgho)
2. [Farhan Sadaf](https://github.com/FarhanSadaf)

## Licensing
The code in this project is licensed under [MIT License](LICENSE).

<!--
## Citation

If you find this work useful, please cite it using the following BibTeX entry:

```bibtex
@misc{das2025enhancedencoderdecoderarchitectureaccurate,
      title={Enhanced Encoder-Decoder Architecture for Accurate Monocular Depth Estimation}, 
      author={Dabbrata Das and Argho Deb Das and Farhan Sadaf},
      year={2025},
      eprint={2410.11610},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.11610}, 
}
-->

