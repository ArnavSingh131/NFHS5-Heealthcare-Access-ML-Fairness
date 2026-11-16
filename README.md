# Predicting Healthcare Access in India: A Machine Learning & Algorithmic Fairness Analysis of NFHS-5 Data

**Course:** UDT108 – Project and Case Study (Semester VII)  
**Institution:** K.R. Mangalam University  
**Author:** Arnav Singh Rathoud  

## Overview
This project develops a supervised machine learning pipeline to predict institutional delivery access (`m15_1`) among women with recent births ($N = 176,843$) using the National Family Health Survey (NFHS-5) Individual Recode microdata. Beyond baseline predictive performance using **Logistic Regression**, **Decision Trees**, and **Random Forest** classifiers balanced via **SMOTE**, this study conducts an ethical **Fairlearn** audit across caste subgroups (`s116`) to uncover systemic predictive disparities.

## Repository Structure
* `NFHS5_Healthcare_Model.ipynb` — End-to-end Jupyter Notebook (data ingestion via `pyreadstat`, preprocessing, SMOTE resampling, model evaluation, and Fairlearn audit).
* `1.png` – `4.png` — Empirical visualizations (EDA wealth/residence access rates, ROC-AUC curves, Random Forest feature importance, and subgroup recall disparity).
* `UDT108_Project_Report.docx` — Complete project documentation.
* `UDT108_Presentation.pptx` — Final external examination presentation deck.

## Setup & Dependencies
1. Obtain the NFHS-5 Individual Recode (`IAIR7EFL.DTA`) from [The DHS Program](https://dhsprogram.com/) (excluded from this repository due to file size and data-use licensing).
2. Install required libraries:
   ```bash
   pip install pandas numpy pyreadstat scikit-learn imbalanced-learn fairlearn matplotlib seaborn
