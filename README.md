# Computer Vision Project

### Plus One Robotics – Round 2

This project presents a feedforward **Convolutional Neural Network (CNN)** based computer vision pipeline designed to learn hierarchical spatial representations from image data, where raw image tensors are transformed through successive convolutional and pooling layers to construct multi-level feature abstractions.

The system performs structured image preprocessing, followed by convolutional feature extraction using learnable kernels, and applies non-linear transformations to generate discriminative feature maps. These extracted representations are then utilized for model training, inference, and performance evaluation, enabling robust visual pattern recognition.


---

## Project Structure

The project is organized into three three modular components, that mirror the processing pipeline implemented in the Jupyter notebook: 

### Part 1: Computer Vision Model (Digit Classification)
This component implements a feedforward convolutional neural network (CNN) for digit classification, trained on single-channel grayscale image tensors. The network performs hierarchical feature extraction through stacked convolutional layers with learnable kernels, non-linear activations, and spatial downsampling via pooling. These operations progressively transform raw pixel intensities into high-level discriminative feature maps, which are then utilized by the classification layers to predict digit classes (0–9).

### Part 2: Sudoku Box Detection
This stage applies classical computer vision techniques to localize and isolate the Sudoku grid from an input image. The pipeline includes image normalization and preprocessing, adaptive thresholding, contour extraction, geometric filtering, and perspective correction to obtain a top-down view of the board. The rectified grid is then segmented into individual cell regions, producing structured inputs suitable for downstream digit recognition.

### Part 3: Solving Sudoku
Detected digits are mapped into a structured 9×9 grid representation corresponding to the Sudoku board state. A constraint-based, rule-driven solving algorithm is then applied to iteratively infer missing values while enforcing Sudoku constraints across rows, columns, and sub-grids. This process continues until a valid and complete solution is obtained.

---

# Dataset information

Kaggle: Sudoku Box Detection
[link text](https://www.kaggle.com/datasets/macfooty/sudoku-box-detection)
* Augmented - Sudoku dataset images.
* 2620 JPEG-Type Images
* 600 x 600 shape


Kaggle: Digits
[link text](https://www.kaggle.com/datasets/karnikakapoor/digits)

* Digits from the Chars74K image dataset
* It contains digits from 0 to 9
* 1016 PNG-type images for each digit
* 128 X 128 shape



## STEP 1: Detailed Breakdown – Digit Classification Pipeline

### 1. Environment Setup and Data Mounting

The notebook is executed in a cloud-based runtime environment where **Google Drive is mounted** to provide persistent access to training and evaluation datasets. This approach enables scalable experimentation and avoids repeated dataset uploads during iterative development.

Image datasets are accessed directly from the mounted drive, ensuring reproducibility and efficient I/O operations during training.

### 2. Dataset Overview

Two publicly available Kaggle datasets are used to support the learning and evaluation stages of the pipeline.

| Dataset Name | Source | Purpose | Image Count | Image Type | Resolution |
|-------------|--------|---------|-------------|------------|------------|
| Sudoku Box Detection | Kaggle | Grid detection and localization support | 2,620 | JPEG | 600 × 600 |
| Digits (Chars74K) | Kaggle | CNN-based digit classification (0–9) | 10,160 (1,016 per digit) | PNG | 128 × 128 |

- **Sudoku Box Detection Dataset**: https://www.kaggle.com/datasets/macfooty/sudoku-box-detection  
- **Digits Dataset (Chars74K)**: https://www.kaggle.com/datasets/karnikakapoor/digits

### 3. Data Preprocessing

Digit images are converted to **single-channel grayscale tensors**, resized to a consistent spatial resolution, and normalized to stabilize gradient-based learning. Labels are encoded to represent digit classes from 0 to 9.

The dataset is split into training and validation subsets to enable performance monitoring and generalization assessment during model training.

### 4. CNN Architecture and Feature Learning

The CNN architecture is designed to progressively learn hierarchical representations:

- Early convolutional layers capture low-level features such as edges and local gradients
- Intermediate layers learn digit-specific strokes and shapes
- Deeper layers encode high-level semantic patterns required for class discrimination

Each convolutional block applies learnable kernels followed by non-linear activation functions and spatial pooling, reducing dimensionality while preserving salient features.

### 5. Training and Evaluation

The model is trained using supervised learning with labeled digit images. During training, the network optimizes classification performance by minimizing loss over digit classes (0–9).

Evaluation metrics are computed on held-out validation data to assess recognition accuracy and robustness before deploying the model for downstream Sudoku grid reconstruction.

