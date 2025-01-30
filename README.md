
# UMKC CS5565 - Intro to Statistical Learning  
**Final Project - FS 2024**

This repository contains the work done for the final project in UMKC's CS5565: **Intro to Statistical Learning** course. The project covers multiple tasks in statistical learning, including regression, classification, feature selection, model optimization, splines, decision trees, support vector machines, and neural networks.

### Objectives:
The primary goal of this project is to apply various statistical learning methods to a dataset of choice and perform model optimization and evaluation. The tasks include:
- **Regression Models**: Linear, Polynomial, Multi-Linear, and other advanced methods like Generalized Additive Models (GAMs) and splines.
- **Feature Selection & Model Optimization**: Techniques like Forward/Backward Stepwise Selection, Principal Component Regression (PCR), and Cross-validation methods.
- **Classification Models**: Logistic Regression, Linear Discriminant Analysis (LDA), Trees, and Support Vector Machines (SVM).
- **Neural Networks**: Image classification using the MNIST dataset.

---

### Datasets Chosen:

The datasets used for this project are:

1. **Breast Cancer Wisconsin (Diagnostic) Dataset**  
   [Kaggle Link](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)  
   - Used for **classification** tasks.
   
2. **Heart Disease UCI Dataset**  
   [Kaggle Link](https://www.kaggle.com/code/imabhilash/heart-disease-uci)  
   - Used for **regression** and **classification** tasks.

3. **Wine Quality Dataset**  
   [Kaggle Link](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009)  
   - Used for **regression** tasks.

---

### Project Structure:

- **Part 1: Regression**
  - **Linear Regression**
  - **Polynomial Regression**
  - **Multi-Linear Regression**

- **Part 2: Feature Selection / Model Optimization**
  - **Forward and Backward Stepwise Selection**
  - **Principal Component Regression (PCR)**
  - **Plots of RSS and Adjusted R²**

- **Part 3: Classification**
  - **Logistic Regression**
  - **Linear Discriminant Analysis (LDA)**
  - **Decision Trees and Random Forest**

- **Part 4: Splines**
  - **Natural and Cubic Splines** (Basis-Splines and Natural Splines)
  - **Polynomial Regression and Step Functions**

- **Part 5: Trees and SVM**
  - **Decision Trees**
  - **Support Vector Machines (SVM)**

- **Part 6: Neural Networks**
  - **Image Classification on MNIST Dataset**
  - **Different topologies and activation functions tested**

---

### Requirements:

1. **Python Libraries**:  
   - `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`, `torchvision`, `keras`, etc.
   
2. **Environment Setup**:
   - To run this code, ensure you have a Python environment set up with the necessary libraries installed.  
   You can set up a virtual environment and install dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Google Colab** is used for training models with **GPU acceleration**.

---

### Project Details:

#### **1. Regression**:
- A variety of regression techniques, including Linear, Polynomial, and Multi-Linear Regression, were implemented using the dataset.
- Each model was evaluated using **train/test split** (70/30 ratio), and results were assessed using **Mean Squared Error (MSE)** and **R-squared** values.
- Plots such as **Residual Sum of Squares (RSS)** and **Adjusted R²** were generated for evaluating model performance.

#### **2. Feature Selection and Model Optimization**:
- **Forward Stepwise** and **Backward Stepwise** selection were performed.
- **Principal Component Regression (PCR)** was applied, and plots of the components against the target variable were created.
  
#### **3. Classification**:
- Models including **Logistic Regression**, **Linear Discriminant Analysis (LDA)**, and **Decision Trees** were trained for classification tasks.
- **Confusion Matrices** and **ROC Curves** were generated for model evaluation.

#### **4. Splines**:
- **Natural and Cubic Splines** were fitted with degrees of freedom of 9, 16, and 22.
- Different polynomial degrees and step functions were also evaluated.

#### **5. Decision Trees and SVM**:
- Decision Trees, Random Forests, and Boosting techniques were applied for both regression and classification tasks.
- **Support Vector Machines (SVM)** were used for multiclass classification with decision boundary plots and confusion matrices.

#### **6. Neural Networks**:
- Neural networks were applied to the **MNIST Image Classification task**.
- Various **topologies** ([256,96,32,10], [96,32,10], [128,64,32,10]) with **sigmoid** and **ReLU** activation functions were tested.
- **Learning rates** and **batch sizes** were optimized for the best performance.

---

### Evaluation Metrics:
- For **regression** tasks: **MSE** and **R-squared** values.
- For **classification** tasks: **Accuracy**, **Precision**, **Recall**, **F1-score**, **Confusion Matrix**, and **ROC Curves**.
- For **neural networks**: **Training Loss**, **Test Loss**, and **Accuracy**.

---

### GitHub Submission:

1. **Completed Notebooks**:  
   - `Regression_Tasks.ipynb`
   - `Feature_Selection_Optimization.ipynb`
   - `Classification_Tasks.ipynb`
   - `Splines_Trees_SVM.ipynb`
   - `Neural_Networks.ipynb`

2. **GitHub Repository Structure**:
   - `/data`: Contains the dataset files or links to Kaggle datasets.
   - `/notebooks`: Contains the Jupyter notebooks for each part of the project.
   - `/images`: Contains relevant plots and images (e.g., ROC, confusion matrices, decision boundary plots).
   - `README.md`: This file with project details.

---

### Final Model Selection:

- The best-performing models were selected based on their ability to generalize well on unseen test data.
- The **Neural Network** with **ReLU activation** and a **learning rate of 0.001** was found to work the best for the **MNIST classification task**.

---
