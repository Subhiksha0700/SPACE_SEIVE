### **SPACE SIEVE**


#### **Team**

| Team Members                   | GitHub ID            |
|--------------------------------|----------------------|
| **Rishi Manohar Manoharan *(POC)***| ***RishiManohar*** |
| Rishikesh Ramesh               | *rishikesh-2809*     |
| Subhiksha Murugesan            | *Subhiksha0700*      |
| Nagul Pandian                  | *CruNt*              |
| Nithish Kumar Senthil Kumar    | *Nithish1201*        |

### **Introduction**

We aim to develop a model that identifies objects in space as active satellites or space debris by analyzing space imagery. This requires differentiating between operational equipment and non-operational pieces or debris that are in orbit.

By integrating two distinct datasets, our goal is to incorporate debris of various sizes to ensure our analysis is resilient to the challenges presented by collisions and the resulting breakup of debris.

It should be of concern to all. Whether it's someone using a smartphone, checking the weather, or navigating with GPS, satellites are integral to these services. However, the increasing clutter of defunct satellites and debris in space poses a significant risk of collision with operational satellites, disrupting these essential services. The addition of more satellites into an already congested space heightens these risks and complicates the expansion of global communication networks. Furthermore, this initiative is crucial in tackling the issue of space debris through strategies like Active Debris Removal (ADR) and the use of Solar Sails.

### **Literature Review**

The literature survey on space debris detection and classification reveals significant advancements and diverse methodologies. The first paper, "Space Debris Detection with Fast Grid-Based Learning," demonstrates a deep learning-based approach emphasizing speed and accuracy in identifying space debris, achieving remarkable results in both synthetic and real data scenarios. The second paper, "Space Objects Classification Techniques: A Survey," provides a comprehensive review of various techniques used for classifying space objects, highlighting the effectiveness of deep learning algorithms, particularly CNNs, in distinguishing between different types of space objects based on light curve data.

Our approach diverges significantly from the methodologies discussed in the surveyed papers by integrating a holistic preprocessing strategy, which includes image resizing for uniformity, data augmentation to enhance variability, uniform image format conversion, and advanced noise reduction techniques to improve image clarity. Unlike the prior works that primarily focus on detection and classification using existing deep learning models, our method employs YOLO for precise object detection and finding bounding box. This allows for better localization and detailed analysis of satellites and debris features. Furthermore, we propose the development of a specialized Convolutional Neural Network tailored specifically for differentiating space debris from functioning satellites, a novel contribution that emphasizes the importance of detailed preprocessing combined with advanced model architecture for improved classification accuracy.

### **Data**

**SPARK2022 AND SPARK2021 Datasets**

The above datasets are an invaluable asset for the field of spacecraft detection and trajectory estimation, hosted by the University of Luxembourg's CVI² at SnT. It provides:

- **URL:** [SPARK2022 Dataset](https://cvi2.uni.lu/spark-2022-dataset) and [SPARK2021 Dataset](https://cvi2.uni.lu/spark-2021-dataset)
- **Content:** Roughly 110,000 high-resolution RGB images.
- **Classes:** 11, including 10 types of spacecraft and one category for space debris.
- **Features:** Images annotated with object bounding boxes and class labels.
- **Conditions:** Simulated under varied space conditions, including extreme scenarios and variable backgrounds, to ensure realism and applicability.

**Data Integrity:** The dataset's unique blend of synthetic and real data, gathered with state-of-the-art simulation technologies and real-world experiments from the Zero-G Lab, ensures high fidelity and applicability to space environments. It has undergone extensive quality assurance processes, including checks for accuracy and consistency.

**Reliability:** Developed in collaboration with leading space agencies, the dataset's authenticity and quality are guaranteed, making it a reliable foundation for machine learning models.

**Usability:** Accompanied by detailed metadata that facilitates a wide range of research applications, from object detection to complex scenario analysis.

The SPARK2022 and SPARK2021 datasets represent a cornerstone in pushing the envelope of what's achievable in space situational awareness and recognition, offering an unparalleled resource for researchers aiming to develop cutting-edge, vision-based algorithms.

### **Methods**

- Two advanced object detection models are under consideration:
    - *Semantic Segmentation using*: Selected Fully Convolutional Networks (FCNs), U-Net, DeepLab, and SegNet for its high precision in object localization tasks.
    - *YOLO (You Only Look Once)*: Selected for its rapid detection capabilities, enabling real-time processing.

### Preprocessing and Data Transformations
1. **Data Ingestion**: The `cv2.imread` function reads the image from the file system. Since OpenCV reads in BGR format, it's converted to RGB using `cv2.cvtColor` to ensure correct color representation.

2. **Resizing**: The `resize_image` function scales images to a uniform size of 640x640 pixels using OpenCV's `cv2.resize`. This standardization simplifies later analysis and model training.

3. **Denoising**: The `denoise_image` function reduces noise via three possible methods:
   - *Non-local Means*: Effective for color images.
   - *Gaussian Blur*: Applies a Gaussian kernel to smooth the image.
   - *Bilateral Filter*: Preserves edges while reducing noise.

4. **Color Space Conversion**: The `convert_color_space` function provides a choice of conversions (HSV, LAB, or grayscale). This operation can highlight or suppress specific color information to aid analysis. The final approach here uses grayscale.

5. **Contrast Enhancement**: The `enhance_contrast` function applies:
   - *CLAHE*: Limited contrast enhancement through a small tile grid to avoid excessive contrast changes.
   - *Histogram Equalization*: Equalizes pixel intensities across an image channel.

6. **Bounding Box Adjustment**: The `adjust_bbox` function rescales bounding box coordinates based on the original and new image dimensions.

7. **Bounding Box Processing**:
   - `process_image_with_bbox` draws the adjusted bounding box around the object of interest and applies a padding of 100 pixels, ensuring no dimension exceeds image boundaries. The cropped image is then returned for further use.

### Model Structure

1. **Data Preparation**: The dataset is loaded from a CSV file into a pandas DataFrame, which is then split into training and validation sets based on a 'usage' flag. Specific images that may cause issues during training or are not needed (e.g., 'image_00006_Debris_2021.jpg') are identified and removed from the training set. The training and validation datasets are shuffled to ensure that the order of data does not affect the learning process.

2. **Image Loading and Processing**:
    - Images are loaded and decoded from JPEG format.
    - Images are resized to a fixed dimension (128x128 pixels) to ensure consistency in input size for the neural network.
    - Pixel values are normalized to the range [0, 1] to aid in convergence during training.

3. **TensorFlow Dataset Creation**: A TensorFlow dataset is created from the list of image paths and labels. This dataset is configured to load and preprocess images in parallel using the `AUTOTUNE` feature, which helps in optimizing the loading process. The dataset is batched (batch size of 64) and prefetched. Prefetching allows the dataset to preload data for the next batch while the current batch is being processed, improving efficiency.

4. **Model Construction**: The model is a convolutional neural network (CNN) that consists of multiple convolutional layers followed by max-pooling layers, dense layers, and dropout layers. Convolutional layers extract features from the images.Pooling layers reduce the dimensionality of the feature maps, thus reducing the number of parameters.Dense layers are fully connected layers that learn non-linear combinations of the high-level features extracted by the convolutional layers. Dropout layers are used to prevent overfitting by randomly setting a fraction of input units to 0 during training. L2 regularization is applied to the dense layers to further prevent overfitting by penalizing large weights.

### Training

1. **Compilation**: The model is compiled with the Adam optimizer (with a custom learning rate of 0.0001), and the binary crossentropy loss function, suitable for binary classification tasks.
   
2. **Execution**: The model is trained on the preprocessed training dataset for 29 epochs, and performance is validated using the validation dataset.

### Evaluation and Prediction

1. **Training History Visualization**: Plots of training and validation accuracy and loss are generated to evaluate the model's performance over epochs.
   
2. **Model Usage**: The trained model is used to predict the class of a new image, demonstrating how the model can be applied to real-world tasks.

This framework ensures uniform and efficient preprocessing of all images, showcasing a typical pipeline for handling image data, constructing a CNN, training the model, and using it for predictions within a machine learning framework tailored for image classification.

### Methods Attempted and Results

1. **Color Space**: Originally tried HSV and LAB, but default image color scale was chosen due to its simplicity and consistency across various objects.
  
2. **Normalization**: The normalization step was omitted in the final approach since it didn't significantly impact the results in this specific task.

3. **Contrast Enhancement**: CLAHE was favored over histogram equalization due to its ability to enhance local regions without overexposing or underexposing.

4. **Denoising**: Bilateral and Gaussian methods provided mixed results, while non-local means proved consistent in retaining details.

### **Results**

1. **Model Evaluation and Outcomes**: To assess the performance of the trained convolutional neural network, we primarily focused on accuracy and loss metrics, as detailed in the training and validation logs. Although accuracy is a straightforward metric, it does not always provide a comprehensive assessment, especially for imbalanced datasets. Hence, it is crucial to consider additional metrics like precision, recall, F1-score.

2. **Training and Validation**: Training Accuracy increased consistently, indicating that the model was effectively learning from the training data. Validation Accuracy also increased but at a slower rate, which is typical as the model tries to generalize on unseen data. Both training and validation Loss decreased over time, suggesting improvements in the model's predictive accuracy as training progressed.

3. **Metrics**: Achieved approximately 93.75% on the training set and 88.05% on the validation set by the 29th epoch. This discrepancy suggests some overfitting despite regularization and dropout. Loss ended at about 0.2096 for training and 0.3316 for validation, indicating the model's good performance in minimizing the classification error.

### Additional Recommended Metrics

For a comprehensive evaluation, especially if the dataset has imbalanced classes or if the costs of false positives and false negatives vary significantly, one should consider:

1. **Precision and Recall**: Precision measures the accuracy of positive predictions. Recall, or sensitivity, measures the ability of the model to find all the relevant cases (all true positives).
   
2. **F1-Score**: The harmonic mean of precision and recall. An F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.

3. **Confusion Matrix**: Would help visualize the performance of the classification model. Each row of the matrix represents the instances in a predicted class, while each column represents the instances in an actual class.

### Goals Achieved

1. **Model Training**: The model was successfully trained with increasing accuracy over time, indicating effective learning from the training data.
2. **Generalization**: The model achieved a reasonable validation accuracy, suggesting it has some ability to generalize to unseen data.
3. **Implementation**: The process involved typical image processing steps, dataset preparation, model construction, and training, which were all implemented correctly.

### **Discussion**

While the primary objectives were met, there are several areas which could be enhanced to achieve more robust results:

1. **Overfitting**: There is a noticeable gap between training and validation accuracy, suggesting overfitting. More sophisticated regularization techniques, data augmentation, or adjustments to the model architecture could help mitigate this.

2. **Evaluation Metrics**: The evaluation was primarily based on accuracy and loss. For a more nuanced understanding of the model's performance, especially in handling different classes, additional metrics like precision, recall, F1-score, and a confusion matrix are necessary. These metrics would help in understanding the balance between sensitivity and specificity, which is crucial for models in practical scenarios.

3. **Model Complexity**: The model architecture might need tuning. For instance, adding or adjusting layers, experimenting with different hyperparameters, or using pre-trained models as a base might yield better results.

Overall, the project succeeded in demonstrating the capability to train a CNN for image classification with good accuracy. However, it somewhat fell short of providing a comprehensive evaluation of the model's performance across various metrics that are crucial for a well-rounded understanding of its practical applicability. Additionally, addressing overfitting more effectively would likely improve the model's ability to generalize, making it more robust in real-world applications.

### **Limitations and Areas for Improvement**

1. **Model Evaluation Metrics**:
   - **Current Limitations**: The project primarily relied on accuracy and loss as metrics. These are basic and often insufficient for a comprehensive understanding, especially in cases where data might be imbalanced.
   - **Improvements**: Incorporating additional metrics such as precision, recall, F1-score, and ROC-AUC would provide a more nuanced evaluation. These metrics would help understand the model's performance in terms of type I and type II errors, which is crucial for applications where the cost of false positives and false negatives is significant.

2. **Handling Overfitting**:
   - **Current Limitations**: There’s evidence of overfitting given the gap between training and validation accuracy. 
   - **Improvements**: Implementing more advanced regularization techniques, such as dropout at different layers or increasing the strength of L2 regularization, could help. Additionally, expanding the dataset through data augmentation techniques or using more complex data splitting strategies like k-fold cross-validation could improve model robustness.

3. **Model Complexity and Architecture**:
   - **Current Limitations**: The model architecture was relatively straightforward. This might limit its ability to capture more complex patterns in data.
   - **Improvements**: Experimenting with more sophisticated architectures, such as deeper networks or architectures like ResNet and Inception. Using transfer learning with pre-trained models could also be explored to leverage learned features from large, diverse datasets.

4. **Dataset Quality and Diversity**:
   - **Current Limitations**: The quality and diversity of the dataset were not discussed, but these are crucial factors. A limited or biased dataset can significantly skew the results.
   - **Improvements**: Enhancing the dataset with more varied examples, including different lighting conditions, angles, and backgrounds, especially if the dataset is used for real-world applications. Additionally, ensuring the dataset is balanced in terms of class representation would improve the model’s ability to generalize.

5. **Real-world Applicability**:
   - **Current Limitations**: It’s unclear how the model performs under real-world conditions, which often present challenges not seen in the training stage.
   - **Improvements**: Conducting field tests and user testing to gather feedback on the model’s performance in practical scenarios. This could lead to iterative improvements in the model based on real-world data and usage patterns.

6. **Scalability and Efficiency**:
   - **Current Limitations**: Training deep learning models can be computationally expensive and time-consuming.
   - **Improvements**: Optimizing the model to improve inference time without compromising accuracy, using techniques like model quantization, pruning, and efficient neural network architectures designed for mobile devices.

Being critical of these aspects is crucial for advancing the project's utility and reliability, particularly when transitioning from a controlled experimental setup to real-world application. Addressing these limitations through systematic testing, validation, and iteration would significantly strengthen the robustness and applicability of the work.

### **Future Works**

1. **Data Representation**:
   - **Goal**: Address discrepancies in the validation dataset by enriching it to better reflect the real-world distribution of data. This aims to reduce the risk of model bias and improve its ability to generalize across more diverse conditions.
   - **Actions**:
     - Collect and integrate more diverse data samples to balance the dataset, particularly focusing on underrepresented classes or scenarios.
     - Conduct a thorough analysis of current dataset biases and gaps, possibly using exploratory data analysis (EDA) to identify areas needing improvement.

2. **Expand to Multiclass**:
   - **Goal**: Transition the model from binary to multiclass classification to make full use of a diverse dataset, allowing for a more detailed and comprehensive analysis.
   - **Actions**:
     - Revise the model architecture to accommodate multiple classes, including adjustments in the output layer and loss function (e.g., using softmax activation and categorical crossentropy).
     - Expand the labeling scheme in the dataset to include multiple classes and ensure that each class is sufficiently represented for training.

3. **Data Representation**:
   - **Goal**: Enhance the model architecture to mitigate overfitting, thus ensuring more reliable performance across various datasets.
   - **Actions**:
     - Implement more sophisticated regularization techniques such as dropout layers or perhaps batch normalization to help the model generalize better.
     - Explore different architectures that might naturally reduce overfitting, such as those involving residual connections or bottleneck layers.

4. **Computing Resources**:
   - **Goal**: Upgrade to high-performance computing resources to fully utilize GPU capabilities, which will facilitate a more sophisticated and efficient model training and evaluation process.
   - **Actions**:
     - Invest in more powerful GPUs or use cloud-based computing resources with better GPU capabilities to reduce training time and enable more complex model experiments.
     - Optimize computing processes, such as improving data pipeline efficiency and model training algorithms, to make full use of the available hardware.
