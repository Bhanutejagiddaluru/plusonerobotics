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

## STEP 1: Detailed Breakdown – Digit Classification Pipeline

### 1. Environment Setup and Data Mounting

The notebook is executed within a cloud-based runtime environment, using Google Collab. Where Google Drive is mounted for storage.


### 2. Dataset Overview

Two publicly available datasets sourced from Kaggle are utilized to support both the learning and perception components of the pipeline.

| Dataset Name | Source | Purpose | Image Count | Image Type | Resolution |
|-------------|--------|---------|-------------|------------|------------|
| Sudoku Box Detection | Kaggle | Grid detection and localization support | 2,620 | JPEG | 600 × 600 |
| Digits (Chars74K) | Kaggle | CNN-based digit classification (0–9) | 10,160 (1,016 per digit) | PNG | 128 × 128 |

- **Sudoku Box Detection Dataset**: https://www.kaggle.com/datasets/macfooty/sudoku-box-detection  
- **Digits Dataset (Chars74K)**: https://www.kaggle.com/datasets/karnikakapoor/digits



### 3. Data Preprocessing

<img width="1244" height="692" alt="image" src="https://github.com/user-attachments/assets/e6d06028-4a4a-4505-80b8-11b4ede6cfcc" />


Digit images are converted to **single-channel grayscale tensors**, resized to a consistent spatial resolution, and normalized to stabilize gradient-based learning. Labels are encoded to represent digit classes from 0 to 9.

The dataset is split into training and validation subsets to enable performance monitoring and generalization assessment during model training.

### 4. CNN Architecture and Feature Learning

The CNN architecture is designed to progressively learn hierarchical representations:

<img width="2140" height="741" alt="image" src="https://github.com/user-attachments/assets/7ad098e8-f1c6-44a3-b4cb-0671d52d7fe6" />

The digit classification model is implemented as a feedforward convolutional neural network using the **TensorFlow Keras Sequential API**. The architecture is explicitly designed to perform progressive spatial feature extraction followed by dense classification, operating on fixed-resolution grayscale image tensors of shape **32 × 32 × 1**.

#### Feature Extraction Stage

The network begins with two convolutional layers, each applying **60 learnable kernels of size 5 × 5** with *same* padding. These capture low-level spatial features such as edges, corners, and intensity gradients. Then **max pooling layer** for downsampling to reduce feature map resolution, the next stage consists of two convolutional layers with **30 kernels of size 3 × 3**, which integrate previously extracted features to learn mid-level abstractions, including digit strokes, contours, and structural compositions, then second **max pooling layer** reduces dimensionality and then followed by  **Dropout layers**, and flatten layer where resulting feature maps into a one-dimensional feature vector, which serves as input to the classification stage.

A fully connected **dense layer with 500 neurons** projects the extracted features into a higher-dimensional latent space, allowing the network to learn complex, non-linear decision boundaries.

The final **softmax output layer** consists of **10 neurons**, corresponding to digit classes (0–9), producing a normalized probability distribution for multi-class classification.

<img width="2030" height="1509" alt="image" src="https://github.com/user-attachments/assets/94e9a90c-65da-4892-8573-409fe286bf59" />

### The CNN model

<img width="760" height="580" alt="image" src="https://github.com/user-attachments/assets/8a407e10-0828-48b9-ac05-77af99979133" />


### 5. Training and Evaluation

The digit classification model is trained using supervised learning on labeled grayscale digit images, where the objective is to learn a mapping from input image tensors to discrete digit classes (0–9). Model training is formulated as a multi-class classification problem and optimized by minimizing categorical cross-entropy loss.

#### Model Compilation and Optimization

The network is compiled using the **RMSprop optimizer**, which is suited for convolutional architectures due to its adaptive learning rate behavior and stable convergence characteristics. The optimizer is configured with a learning rate of **0.001**, momentum term (**ρ = 0.9**), and numerical stability constant (**ε = 1e−8**). Model performance during training is tracked using **classification accuracy** as the primary evaluation metric.

#### Training Procedure

Training is performed over **30 epochs** using mini-batch gradient descent with a **batch size of 32**. To improve generalization and reduce overfitting, **data augmentation** is applied via an ImageDataGenerator, Each epoch consists of **200 training steps**, with validation performance evaluated on a held-out dataset at the end of every epoch.

#### Evaluation and Model Validation

After training, the model is evaluated on an independent test dataset to assess its performance. The evaluation reports both the final loss value and classification accuracy, providing a quantitative measure of the network’s robustness prior to deployment in the downstream Sudoku grid reconstruction and digit inference pipeline.

---

## Part 2: Sudoku Grid Detection and Perspective Normalization

This stage implements a **computer vision–based perception pipeline** to localize, extract, and geometrically normalize the Sudoku board from an input image prior to digit recognition.

### 1. Input Selection and Standardization

A Sudoku image is randomly sampled from the dataset. The selected image is resized to a fixed spatial resolution of **450 × 450 pixels**

### 2. Image Preprocessing and Binarization

The input image undergoes a structured preprocessing pipeline to enhance grid visibility and reduce noise:

- The image is converted to **grayscale**, reducing dimensionality while preserving structural information.  
- **Gaussian blurring** is applied to attenuate high-frequency noise and improve threshold stability.  
- **Adaptive thresholding** is used to generate a binary representation of the image, enabling separation of grid lines and digits under varying illumination conditions.  

This preprocessing stage produces a high-contrast binary image suitable for contour-based analysis.

### 3. Contour Extraction and Grid Localization

Contours are extracted from the thresholded image using **external contour retrieval**, isolating the outermost boundaries present in the scene. Each detected contour is evaluated based on:

- **Contour area**, to filter out noise and small artifacts  
- **Polygonal approximation**, to identify quadrilateral shapes  

The algorithm selects the **largest four-sided contour**, corresponding to the outer boundary of the Sudoku grid. This contour is assumed to represent the board due to its dominant area and rectangular geometry.

### 4. Corner Reordering and Geometric Consistency

Once the grid contour is identified, its four corner points are reordered into a canonical orientation (**top-left, top-right, bottom-left, bottom-right**). This step ensures geometric consistency and is critical for accurate perspective correction.

Point ordering is computed using **sum and difference heuristics** over the corner coordinates, enabling reliable spatial alignment regardless of grid orientation in the input image.

### 5. Perspective Transformation and Board Rectification

A **homography matrix** is computed using the detected grid corners and a predefined rectangular target plane. This transformation enables perspective normalization, producing a **top-down, axis-aligned view** of the Sudoku board.

The warped image represents a geometrically rectified grid with uniform cell dimensions, eliminating perspective distortion introduced during image capture.

### 6. Grid Extraction and Preparation for Digit Recognition

The rectified Sudoku image is converted to grayscale and serves as the input for grid segmentation. The image is conceptually divided into a **9 × 9 lattice**, yielding **81 individual cell regions**, each corresponding to a single Sudoku cell.

These extracted cell images are subsequently forwarded to the **CNN-based digit classification module** for numerical inference.

---

## Part 3: Digit Inference, Grid Reconstruction, and Sudoku Solving

This stage integrates learned perception with symbolic reasoning to perform end-to-end Sudoku puzzle solving, transforming visual inputs into a valid numerical solution.

### 1. Puzzle Acquisition and Grid Normalization

A Sudoku puzzle image is loaded and resized to a fixed resolution of **450 × 450 pixels** to maintain consistency with the grid detection pipeline. The image undergoes grayscale conversion, Gaussian smoothing, and adaptive thresholding to ensure robustness against illumination and noise variations.

Using contour detection and geometric filtering, the outer Sudoku grid is identified, reframed, and perspective-normalized via a **homography transformation**, producing a top-down, axis-aligned representation of the board suitable for structured cell extraction.

### 2. Cell Segmentation and Spatial Cropping

The rectified Sudoku image is divided into a **9 × 9 lattice**, yielding **81 individual cell regions** corresponding to the board layout. Each cell is spatially cropped to remove grid borders and peripheral artifacts, isolating the region most likely to contain a digit.

This cropping step improves the signal-to-noise ratio by eliminating irrelevant pixels and ensuring that the digit occupies the central receptive region expected by the CNN classifier.

### 3. Digit Recognition via CNN Inference

Each cropped cell image undergoes a standardized preprocessing pipeline prior to inference:

- Conversion to a NumPy tensor  
- Border trimming to remove residual grid artifacts  
- Resizing to **32 × 32 pixels**, matching the CNN input specification  
- Pixel-wise normalization to the range **[0, 1]**

The preprocessed cell images are batched and passed through the trained CNN model, which outputs a probability distribution over digit classes (0–9) for each cell.

A confidence threshold is applied to the softmax outputs to ensure prediction reliability. Cells with maximum class probability below the threshold are classified as empty, while confident predictions are mapped to their corresponding digit values.

### 4. Grid Reconstruction

The predicted digit sequence is reshaped into a **9 × 9 numerical matrix**, forming a structured representation of the Sudoku board state. Empty or uncertain cells are explicitly encoded as zeros, preserving compatibility with downstream constraint-solving logic.

This reconstructed grid serves as the interface between perception (computer vision and CNN inference) and reasoning (Sudoku solving).

### 5. Constraint-Based Sudoku Solving

The reconstructed grid is solved using a **recursive backtracking algorithm**, a depth-first search strategy commonly used for constraint satisfaction problems.

The solver operates as follows:

- Identify the next empty cell in the grid  
- Iteratively test candidate values (1–9)  
- Validate each candidate against Sudoku constraints:
  - Row consistency  
  - Column consistency  
  - 3 × 3 sub-grid consistency  
- Recursively propagate valid assignments until a complete solution is found or all possibilities are exhausted  

If a valid terminal state with no remaining empty cells is reached, the puzzle is deemed solved. If no valid configuration exists, the failure is attributed to digit misclassification or ambiguous perception output.

### 6. Solution Output and Validation

Once solved, the final Sudoku grid is printed in a structured, human-readable format with clear sub-grid separators. This output represents the end-to-end result of the integrated **perception–reasoning pipeline**.


---

Thank You



