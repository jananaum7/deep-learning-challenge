# Deep Learning Challenge 

## Table of Contents
1. [Getting Started](#getting_started)
2. [Introduction](#introduction)
3. [Alphabet Soup Charity Classification Report](#alphabet_soup_report)
4. [Contributers](#contributers)

## Getting Started
   - Clone the Repository: Open Git Bash and navigate to the directory where you want to store the project. Use the following command to clone the repository:
     - ``` bash
         git clone git@github.com:jananaum7/deep-learning-challenge.git
        ```
   - Navigate to the Project Directory: Change into the newly created project directory:
     - ``` bash
       cd deep-learning-challenge
       ```
   - Create the Starter_code Folder:
     - ``` bash
       mkdir Starter_code
       ```
   - Add Starter Code and Resources

## Introduction
In this analysis, I created a deep learning model to predict the success of charity organizations in securing funding. The goal was to determine whether a charity would be considered "successful" (1) or "unsuccessful" (0) in reaching its fundraising goals, based on various characteristics. By using this model, charities can better understand the factors that influence their chances of success and use the insights to improve their fundraising efforts.

## Alphabet Soup Charity Classification Report
   - Purpose of the Analysis:
     - The main objective of this analysis was to create a classification model to predict whether a charity would succeed in receiving funds based on its features. By analyzing various attributes, the model helps charities understand the key factors that contribute 
       to their success, ultimately supporting more effective decision-making in fundraising efforts.
   - Target Variable Overview:
     - The target variable, or the outcome we are trying to predict, is whether a charity is "successful" (1) or "unsuccessful" (0) in securing funding. The dataset includes several features that provide information about each charity's characteristics, such as its 
       budget, category, and audience engagement. The success of a charity in securing funding is the primary focus of this model.
   - Data Preprocessing:
     - Target Variable(s):
       - The target variable is the IS_SUCCESSFUL column, which indicates whether the charity received funding (1) or not (0). This is a binary classification problem.
       - Features:
         - The features consist of various characteristics of the charity, such as APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and others. These features provide important information that helps the model predict whether a charity 
           will be successful in securing funding.
       - Variables to Remove:
         - Some variables like EIN and NAME are not relevant for prediction purposes and should be removed from the dataset. These columns do not provide meaningful information that can help predict the charity’s success or failure in securing funding.  
  - Compiling, Training, and Evaluating the Model
    - Model Architecture:
      - Neurons:
        - I chose 80 neurons for the first hidden layer and 30 for the second. This decision was made to allow the model to learn more complex patterns and relationships in the data, improving its ability to predict charity success.
      - Layers:
        - The model consists of three layers: an input layer with 43 features, two hidden layers with 80 and 30 neurons, and an output layer with a single neuron for binary classification (successful or unsuccessful).
      - Activation Functions:
        - For the hidden layers, I used the ReLU activation function, which helps the model learn non-linear relationships within the data. For the output layer, I used the sigmoid activation function, which is ideal for binary classification tasks 
          as it produces outputs between 0 and 1, representing the likelihood of success.
    - Model Performance:
      - The goal was to achieve a classification accuracy of at least 80%. After training, the model successfully met this target, showing solid performance in distinguishing between charities that are likely to succeed and those that are not.
    - Steps Taken to Improve Model Performance:
      - Data Cleaning:
        - Dropped non-beneficial columns like EIN and NAME from the dataset to avoid unnecessary noise in the model.
      - Feature Simplification:
        - Examined unique values in categorical columns like APPLICATION_TYPE and CLASSIFICATION.
        - Grouped low-frequency categories in APPLICATION_TYPE and CLASSIFICATION into an "Other" category to reduce the sparsity and improve generalization.
      - Preprocessing:
        - It seems like you prepared the data for machine learning, possibly including normalization or standardization (the use of StandardScaler was imported, but not fully analyzed in the snippet).
      - Data Splitting:
        - Used train_test_split from Scikit-learn, ensuring proper partitioning of the dataset into training and testing subsets to evaluate performance.
- Summary
  - The notebook focuses on building a binary classification model with data preprocessing steps like dropping irrelevant columns, grouping low-frequency categories, encoding categorical variables, and scaling features. A TensorFlow neural network with a hidden layer 
    using ReLU activation, and a sigmoid output layer is trained using the Adam optimizer and binary crossentropy loss. The model is evaluated on test data, showcasing a systematic approach to improving predictive performance.

## Contributers
 - J. Naum
 - Minor error corrections and README improvements were assisted by [ChatGPT](https://openai.com/chatgpt) and Xpert learning assistant. 
