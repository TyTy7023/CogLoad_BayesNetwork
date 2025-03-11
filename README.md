### **Cog_BayesNetwork**  

## **About My Data**  
- Includes three directories (**23_objects**, **allFeatures**, **last_30s_segments**):  
  + **23_objects**: Raw data from 23 subjects recorded during data collection.  
  + **allFeatures**: Contains **statFeatures.csv** and **features.csv**.  
    - **statFeatures.csv**: Includes 10 basic statistical features (referenced from the original paper: [Colab_part-1](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)), such as mean, standard deviation, skewness, kurtosis, diff, diff², 25th quantile, 75th quantile, qdev, and max-min.  
    - **allFeatures.csv**: Contains both basic statistical features and expert features (referenced from the original paper: [Colab_part-2](https://colab.research.google.com/drive/1adYKWqgSsky0z5LITB9QjsFTmL7g90gH?usp=sharing)).  
  + **last_30s_segments** (**Main working dataset**): Includes physiological signals (**GSR, HR, RR, Temp**) extracted from the last 30 seconds of each signal along with label information (**Rest/Load**).  
- After preprocessing, the data was adjusted to fit the **Bayesian Network model**, and all datasets were uploaded to **[DATASET](https://www.kaggle.com/datasets/quanminhminhquan/cognitiveload)**.  

## **Purpose**  
- Develop a **Bayesian Network model** based on selected features to improve the classification performance of cognitive load signals. The study focuses on exploring the relationships between physiological signals (**RR, GSR, HR, Temp**) and cognitive load states (**Rest/Load**) using a Bayesian model.  

<div style="text-align: center;">
  <img src="https://learningpartnership.s3.amazonaws.com/uploads/asset_image/2_299.jpg" alt="CognitiveLoad" width="400" style="text-align: left;"/>
  <img src="img/signal.png" alt="Signal" width="335" style="text-align: right;"/>
</div>  

## **Research Approach**  
  + **Analysis and evaluation of existing methods**: Examine and assess traditional models such as **SVM, Random Forest, Gradient Boosting, CNN, RNN, etc.**, to identify optimal methods for cognitive load signal classification.  
  + **Developing a new approach**: Propose a **Bayesian Network model**, where physiological signals (**RR, GSR, HR, Temp**) act as **parent nodes**, extracted features are **child nodes**, and the classification labels (**Rest/Load**) represent the **output**.  
  + **Experimentation and evaluation**: Evaluate the performance of the **Bayesian Network model** to determine its applicability in cognitive load classification.  

## **Result**  
