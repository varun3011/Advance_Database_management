# Lyft Ride Reviews Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" alt="Apache Spark">
</p>

## 🎯 Project Overview
This project aims to analyze Lyft ride reviews to extract meaningful insights through sentiment analysis using machine learning techniques. It classifies reviews into positive and negative sentiments and provides visualizations of trends and predictions.

---

## 🚀 Features

- **Sentiment Classification**: Classify customer reviews as positive or negative using logistic regression.
- **Data Cleaning**: Preprocessing reviews by removing neutral ratings and stop words.
- **Visualization**: Generate visual insights like logistic regression coefficients and ROC curves.
- **Documentation**: Complete guidance on setting up and running the project.

---

## 🛠️ Tools & Technologies

- **Python 3.11**
- **Apache Spark**
- **Matplotlib**
- **Pandas**
- **NumPy**

---

## 🏗️ Architecture & Workflow

1. **Data Loading**: Load Lyft reviews from CSV files.
2. **Data Cleaning**: Remove neutral ratings and irrelevant stop words.
3. **Feature Engineering**: Convert text data into numerical features using `CountVectorizer`.
4. **Model Training**: Train a logistic regression model to classify sentiments.
5. **Visualization**: Plot logistic regression coefficients and ROC curves.
6. **Evaluation**: Evaluate the model's accuracy using AUC scores.

---

## 📈 Visual Outputs

### Logistic Regression Coefficients

```python
plt.plot(beta)
plt.title('Logistic Regression Coefficients (Beta)')
plt.show()
```

### ROC Curve

```python
plt.plot(false_positive_rate, true_positive_rate, color='#8cbfd0', linestyle='-', linewidth=3.)
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()
```

---

## 📋 How to Run the Application

### 1️⃣ Prerequisites

- Python 3.11 or above
- Required Python packages:
  ```bash
  pip install pandas numpy matplotlib pyspark
  ```

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/your-repo/lyft-reviews-sentiment-analysis.git
cd lyft-reviews-sentiment-analysis
```

### 3️⃣ Run the Application

1. Load the dataset by placing `Lyft_Ride_Reviews_Augmented_for_User.csv` in the working directory.
2. Execute the script:

```bash
python analysis.py
```

3. View the outputs:
   - Logistic regression coefficients graph
   - ROC curve
   - Sentiment predictions for the test dataset

---

## 📂 Repository Structure

```plaintext
lyft-reviews-sentiment-analysis/
├── analysis.py         # Main analysis script
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation
└── Lyft_Ride_Reviews_Augmented_for_User.csv # Dataset
```
