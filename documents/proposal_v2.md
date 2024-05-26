## **SPACE SIEVE** (Using types of R-CNN Machine Learning Techniques to classify space debris from functioning satellites)


### **Team**

| Team Members                   | GitHub ID            |
|--------------------------------|----------------------|
| **Rishi Manohar Manoharan *(POC)***| ***RishiManohar*** |
| Rishikesh Ramesh               | *rishikesh-2809*     |
| Subhiksha Murugesan            | *Subhiksha0700*      |
| Nagul Pandian                  | *CruNt*              |
| Nithish Kumar Senthil Kumar    | *Nithish1201*        |

### **Introduction**

We aim to develop a model that automatically identifies objects in space as active satellites or space debris by analyzing space imagery. This requires differentiating between operational equipment and non-operational pieces or debris that are in orbit.

By integrating two distinct datasets, our goal is to incorporate debris of various sizes to ensure our analysis is resilient to the challenges presented by collisions and the resulting breakup of debris.

It should be of concern to all. Whether it's someone using a smartphone, checking the weather, or navigating with GPS, satellites are integral to these services. However, the increasing clutter of defunct satellites and debris in space poses a significant risk of collision with operational satellites, disrupting these essential services. The addition of more satellites into an already congested space heightens these risks and complicates the expansion of global communication networks. Furthermore, this initiative is crucial in tackling the issue of space debris through strategies like Active Debris Removal (ADR) and the use of Solar Sails.

### **Literature Review**

The literature survey on space debris detection and classification reveals significant advancements and diverse methodologies. The first paper, "Space Debris Detection with Fast Grid-Based Learning," demonstrates a deep learning-based approach emphasizing speed and accuracy in identifying space debris, achieving remarkable results in both synthetic and real data scenarios. The second paper, "Space Objects Classification Techniques: A Survey," provides a comprehensive review of various techniques used for classifying space objects, highlighting the effectiveness of deep learning algorithms, particularly CNNs, in distinguishing between different types of space objects based on light curve data.

Our approach diverges significantly from the methodologies discussed in the surveyed papers by integrating a holistic preprocessing strategy, which includes image resizing for uniformity, data augmentation to enhance variability, uniform image format conversion, and advanced noise reduction techniques to improve image clarity. Unlike the prior works that primarily focus on detection and classification using existing deep learning models, our method employs Mask-RCNN for precise object detection and segmentation. This allows for better localization and detailed analysis of satellites and debris features. Furthermore, we propose the development of a specialized Convolutional Neural Network tailored specifically for differentiating space debris from functioning satellites, a novel contribution that emphasizes the importance of detailed preprocessing combined with advanced model architecture for improved classification accuracy.


### Stakeholders
1.⁠ **⁠Space Agencies (NASA, SpaceX):** Engage in a variety of space exploration, satellite deployment, and research missions. Effective debris management is vital to protect their investments and ensure the success and safety of their missions.Interested in monitoring and managing space debris to ensure the safety and longevity of satellite operations.

2.⁠ **⁠Telecommunication Companies:** Rely on satellite networks for communications; effective debris management is crucial to maintain service continuity and safeguard space infrastructure.


3.**Spacecraft Raw Material Producers:** Supply materials for satellite manufacturing. They are impacted by the demand for more durable and debris-resistant satellite designs, influencing material innovation and production processes.

4.⁠ **⁠Government:** Invests in space infrastructure and seeks to minimize financial losses due to debris-related damages.

Our work aims to provide these stakeholders with more precise, reliable, and efficient tools for identifying, classifying, and managing space objects, thereby enhancing the overall safety and sustainability of space operations.


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

Our preprocessing involves resizing images for uniformity, augmenting data to introduce variety, converting all image files to a uniform format and applying noise reduction techniques for clarity. 

A key step is using Mask-RCNN for object detection and image segmentation, enabling precise localization and outlining of satellites and debris, essential for analyzing their distinct features.

The preprocessed data is passed to self-deployed Convolutional Neural Network to classify space debris from functioning satellites. Our evaluation metrics will focus on precision, recall, F1 scores to ensure our model's predictions are not only accurate but also reliable.

### **Project Plan**

| Period                                     | Milestone                                  |
|--------------------------------------------|--------------------------------------------|
| 27th February 2024 (Checkpoint 1)          | Proposal                                   |
| 25th March 2024 (Checkpoint 2)             | All preprocessing steps including Mask RCNN|
| 22nd April 2024 (Checkpoint 3)             | Final modelling with CNN                   |
| 1st May 2024                               | Final Report Submission.                   |



### **Risks**

**Misclassification of Miniature Satellites**

- **Risk:** Small satellites, such as CubeSats, could be misclassified as space debris due to their diminutive dimensions. Specifically, a CubeSat is a compact, cube-shaped satellite, each side measuring a mere 10 centimeters.

- **Mitigation:** Augment the dataset with a diverse array of miniature satellite images and incorporate insights from subject matter experts to refine the differentiation process.

**RCNN Detection Failures**

- **Risk:** If RCNN fails to detect objects, it could impair the CNN's subsequent analysis.

- **Mitigation:** Implement a multi-algorithm strategy for object detection to provide a backup if RCNN underperforms, ensuring continuous analysis flow.

**Augmented Data Misclassification**

- **Risk:** Enhanced datasets could generate non-representative images, potentially leading to inaccurate classification results.

- **Mitigation:** By setting clear boundaries for data augmentation to maintain image integrity and periodically review the model performance on augmented versus original data to ensure robustness.


