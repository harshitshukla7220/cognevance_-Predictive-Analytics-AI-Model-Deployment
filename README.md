# Predictive Analytics & AI Model Deployment
## Customer Churn Prediction using Machine Learning

**Internship Project – Cognevance**

[GitHub Repository](https://github.com/harshitshukla7220/cognevance_customer-churn-predictive-analytics)

---

## 1. Project Overview

The **Predictive Analytics & AI Model Deployment** project aims to develop a complete machine learning pipeline that predicts customer churn using historical telecommunications customer data.

Customer churn occurs when customers stop using a company's products or services. This project uses machine learning algorithms to identify customers who are likely to leave, helping businesses make better customer retention decisions.

The project covers the complete machine learning lifecycle, including data collection, preprocessing, feature engineering, model training, evaluation, deployment, visualization and documentation.

## 2. Project Objectives

- Analyze customer data to identify churn patterns.
- Perform data cleaning and advanced preprocessing.
- Apply feature engineering techniques.
- Train and compare multiple machine learning algorithms.
- Evaluate models using different performance metrics.
- Predict customer churn probability.
- Deploy the trained model using FastAPI.
- Develop an interactive Streamlit dashboard.
- Generate evaluation reports and visualizations.

## 3. Dataset Description

**Dataset:** Telco Customer Churn Dataset

- Total records: 7,043
- Total columns: 21
- Target variable: `Churn`
- Prediction classes: Yes / No
- Domain: Telecommunications and Business Analytics

The dataset contains information about customer demographics, services, contracts, tenure and billing details.

### Target distribution

| Customer category | Records |
|---|---:|
| No Churn | 5,174 |
| Churn | 1,869 |
| Total | 7,043 |

The target variable is used to predict whether a customer is likely to leave the service.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Joblib | Model serialization |
| FastAPI | Model deployment and REST API |
| Streamlit | Interactive dashboard |
| Git & GitHub | Version control |
| Docker | Containerization |
| Pytest | Automated testing |

## 5. Machine Learning Algorithms

The project implements and compares five machine learning algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Extra Trees Classifier
4. Gradient Boosting Classifier
5. Multi-Layer Perceptron (MLP) Neural Network

The best-performing model is selected using cross-validation PR-AUC.

## 6. Data Preprocessing & Feature Engineering

The preprocessing pipeline includes:

- Removing customer ID columns.
- Converting numerical columns into appropriate data types.
- Handling missing values using median and most-frequent imputation.
- Encoding categorical variables using One-Hot Encoding.
- Standardizing numerical features.
- Creating additional predictive features.

### Engineered features

- Customer tenure in years.
- Customer tenure groups.
- Average monthly charges.
- Number of subscribed services.
- Month-to-month contract indicator.
- Electronic check payment indicator.
- Internet service indicator.

## 7. Model Evaluation

The models are evaluated using:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

### Actual model performance

The following results were obtained from the held-out test dataset:

| Metric | Result |
|---|---:|
| Selected Model | Logistic Regression |
| Accuracy | 76.79% |
| Precision | 54.65% |
| Recall | 73.80% |
| F1-score | 62.80% |
| ROC-AUC | 84.25% |
| PR-AUC | 63.71% |

**Note:** These results are dataset-specific and are not a guarantee of future production performance.

## 8. Predictive Analytics

The trained model predicts the probability of customer churn.

The system provides:

- Customer churn classification.
- Churn probability.
- Customer retention probability.
- Configurable decision threshold.

Predictions can help businesses identify customers who may require retention support.

## 9. Model Deployment

The trained model is deployed using FastAPI.

### API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | API welcome page |
| `/health` | GET | API health status |
| `/model-info` | GET | Model information |
| `/predict` | POST | Individual prediction |
| `/predict/batch` | POST | Batch predictions |
| `/docs` | GET | Swagger API documentation |

The API accepts customer information and returns churn predictions and probabilities.

## 10. Interactive Dashboard

The Streamlit dashboard provides:

- Customer churn distribution.
- Dataset overview.
- Model comparison.
- Performance metrics.
- Confusion matrix visualization.
- Individual customer churn prediction.
- Downloadable evaluation reports.

## 11. Project Structure

```text
customer-churn-predictive-analytics/
│
├── data/
│   ├── customer_churn.csv
│   └── DATASET_SOURCE.md
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── train.py
│   ├── inference.py
│   ├── app.py
│   └── data_audit.py
│
├── dashboard/
│   └── dashboard.py
│
├── models/
│   └── best_model.joblib
│
├── reports/
│   ├── model_metrics.json
│   ├── model_report.md
│   └── data_quality_report.json
│
├── images/
│   └── confusion_matrix.png
│
├── tests/
│
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
└── docker-compose.yml
```

## 12. Installation and Setup

### Requirements

- Python 3.11 or newer
- Visual Studio Code
- Git
- Internet connection for dependency installation

### Step 1: Clone the repository

```bash
git clone https://github.com/harshitshukla7220/cognevance_customer-churn-predictive-analytics.git
```

### Step 2: Open project folder

```bash
cd cognevance_customer-churn-predictive-analytics
```

### Step 3: Create virtual environment

```bash
py -3.13 -m venv .venv
```

### Step 4: Activate virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Step 6: Generate data audit report

```bash
python -m src.data_audit
```

### Step 7: Train machine learning models

```bash
python -m src.train
```

The training process generates the trained model and evaluation reports.

### Step 8: Run automated tests

```bash
pytest -q
```

## 13. Run FastAPI

```bash
uvicorn src.app:app --reload
```

Open the following URL:

http://127.0.0.1:8000/docs

## 14. Run Streamlit Dashboard

Open another terminal and activate the virtual environment.

```bash
streamlit run dashboard/dashboard.py
```

Open:

http://localhost:8501

## 15. Docker Deployment

If Docker Desktop is installed:

```bash
docker compose up --build
```

API: http://localhost:8000/docs

Dashboard: http://localhost:8501

## 16. Expected Outputs

After successful execution, the project generates:

- Trained machine learning model.
- Model comparison results.
- Evaluation metrics in JSON format.
- Detailed Markdown evaluation report.
- Data quality report.
- Confusion matrix visualization.
- Individual and batch prediction results.
- Interactive analytics dashboard.
- REST API for model inference.

## 17. Future Enhancements

- Integrate additional real-world datasets.
- Experiment with TensorFlow and advanced neural networks.
- Improve model performance through hyperparameter optimization.
- Add cloud deployment using AWS, Azure or Render.
- Implement real-time customer churn monitoring.
- Add model drift and fairness monitoring.

## 18. Conclusion

This project demonstrates a complete machine learning workflow for customer churn prediction, from data preprocessing and model training to evaluation and deployment.

It combines multiple machine learning algorithms, predictive analytics, FastAPI and Streamlit to create a practical business intelligence solution.

The project provides hands-on experience with machine learning, model evaluation, API development, data visualization and deployment.

---

## Author

**Harshit Shukla**

B.Tech – Information Technology

**Internship Organization:** Cognevance

**GitHub:** https://github.com/harshitshukla7220

**Project Repository:** https://github.com/harshitshukla7220/cognevance_customer-churn-predictive-analytics

---

*Developed as part of the Cognevance Internship Project – Predictive Analytics & AI Model Deployment.*
