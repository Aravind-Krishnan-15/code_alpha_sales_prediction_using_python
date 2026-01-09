# Sales Prediction Using Python

## Project Overview
This project focuses on **predicting future sales** using Python and Machine Learning techniques.  
Sales are forecasted based on **advertising spend, target customer segment, and marketing platform**.  
The model also analyzes how changes in advertising budget impact sales outcomes and provides **actionable business insights**.

---

## Objectives
- Predict future sales using historical data  
- Understand the impact of advertising spend on sales  
- Identify effective marketing platforms and target segments  
- Support data-driven marketing strategies  

---

## Technologies Used
- **Python**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computations
- **Matplotlib** – Data visualization
- **Scikit-learn** – Machine Learning (Linear Regression)

---

## Dataset Description
The dataset includes the following features:

| Column Name           | Description |
|----------------------|-------------|
| Advertising_Spend    | Money spent on advertising |
| Target_Segment       | Customer segment (Youth, Adult, Senior) |
| Platform             | Advertising platform (Online, TV, Social Media) |
| Sales                | Sales revenue generated |

A sample dataset is created directly in the code for simplicity.

---

## Project Workflow
1. **Data Creation & Loading**
2. **Data Cleaning**
3. **Data Transformation**
   - One-hot encoding for categorical variables
4. **Feature Selection**
5. **Train-Test Split**
6. **Model Training (Linear Regression)**
7. **Sales Prediction**
8. **Model Evaluation**
9. **Advertising Impact Analysis**
10. **Future Sales Forecasting**

---

## Model Used
- **Linear Regression**
  - Suitable for understanding the relationship between advertising spend and sales
  - Easy to interpret and effective for small to medium datasets

---

## Model Evaluation Metrics
- **Mean Squared Error (MSE)**
- **R² Score**

These metrics help evaluate the accuracy and reliability of the prediction model.

---

## Advertising Impact Analysis
A scatter plot is used to visualize the relationship between:
- **Advertising Spend**
- **Sales Revenue**

This helps businesses understand how increasing or decreasing ad spend affects sales.

---

## Future Sales Prediction
The model predicts sales for new advertising scenarios by:
- Ensuring feature columns match the trained model
- Using optimized advertising inputs

Example:
```python
Advertising_Spend = 4500
Target_Segment = Youth
Platform = Social Media
