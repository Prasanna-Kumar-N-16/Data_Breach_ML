# Cyber Security Breaches Analysis

This repository contains a code which performs an analysis of cyber security breaches dataset.

## Overview

The code is structured to provide a comprehensive analysis of the cyber security breaches dataset. It includes data preprocessing, exploratory data analysis, feature transformation, and predictive modeling.

## Usage

### Prerequisites

- Python 3.x
- Any python IDE

### Installation

1. Clone this repository:

    ```
    git clone https://github.com/Prasanna-Kumar-N-16/Data_Breach_ML.git
    ```

2. Navigate to the project directory:

    ```
    cd Data_Breach_ML
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

### Running the Notebook

1. Open IDE and navigate to the repo directory:

    ```
    python3 main.py
    ```


## Contents

1. **Introduction**: 
       Data breaches have become a usual concern in today's digital world, posing significant
       threats to organizations across various sectors. In this study, we delve into a comprehensive
       dataset sourced from diverse academic outlets, governmental disclosures, and mainstream
       publications, capturing incidents involving compromised records. While acknowledging the
       persistence of smaller-scale breaches, our focus remains on the broader landscape, where
       hacking emerges as a predominant method of compromise.
       The implications of data breaches extend beyond mere statistics, profoundly impacting
       organizations of all sizes and sectors. As such, understanding the underlying patterns of
       vulnerability is imperative. Through this research, we aim to contribute to scholarly discourse
       by these patterns, identifying the most vulnerable entities and the techniques employed by
       attackers

2. **Dataset**: 
       The dataset consists of 7 columns and 352 rows , each providing crucial information about
       data breaches. Here's a breakdown of the columns and their descriptions:

       **1.ID:** This column is the key to each data breach . It is represented as an int, allowing for
                 unique identification of the breached entities.
       **2.Entity:** This column contains the names of the organizations that experienced data
                 breaches. It is represented as a string, allowing for easy identification of the breached entities.
       **3.Year:** This column indicates the year in which each data breach occurred. It is represented
                 as an integer, providing a chronological understanding of when the breaches took place.
       **4.Records:** This column quantifies the number of records compromised in each breach. It is
                 represented as an integer, offering insight into the scale of the breaches.
       **5.Organization type:** This column categorises the type of organisations that were breached.
                 It is represented as a string, allowing for classification based on organisational characteristics.
       **6.Method:** This column details the method used to breach each organisation. It is
                 represented as a string, providing information on the techniques employed by attackers.
       **7.Sources:** This column indicates the sources from which the data regarding the breaches
                 was collected. It is represented as a string, offering transparency regarding the origin of the
                 information

3. **Data Preprocessing**: 
              Remove Sources Column ,  Drop Irrelevant records .

4. **Feature Transformation**: 
              Label Encoding , Principal Component Analysis (PCA) .

5. **Training Methods**:

             Decision Tree Regression (DTRegression)
             Random Forest Regression (RandomForest)
             XGBoost Regression (XGBoosting)
             Neural Network Regression (using Keras Sequential model)

6. **Results and Conclusion**: 

           In summary we tackle the task of predicting the number of records in cyber-security breaches
           using various regression models. Initially, we do preprocessing and feature transformation to
           ensure data quality and compatibility with the models. Through statistical tests and
           correlation analyses, I determined that the "Method" and "Organization type" features hold
           significant predictive power. These insights help with the feature selection process, focusing
           model training efforts on the most relevant aspects of the data.

