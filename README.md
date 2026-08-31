# Mobile Money Fraud Detector

## 1. Project Overview

The Mobile Money Fraud Detector is an AI-powered machine learning application designed to identify potentially fraudulent mobile money transactions.

The system analyzes transaction characteristics such as transaction amount, transaction type, account balance, transaction frequency, account age, transaction time, device changes, and location changes. A machine learning model then assigns a fraud risk and classifies the transaction as either legitimate or potentially fraudulent.

## 2. Problem Statement

The rapid growth of mobile money and digital financial transactions has created increased opportunities for financial fraud. Fraudulent transactions can result in financial losses, reduced customer trust, and security risks for financial service providers.

Traditional rule-based fraud detection systems may struggle to identify complex and changing transaction patterns. This project demonstrates how machine learning can be used to identify suspicious transaction behavior automatically.

## 3. Objectives

The objectives of the project are to:

1. Develop a machine learning model for mobile money fraud detection.
2. Identify transaction patterns associated with fraudulent behavior.
3. Classify transactions as legitimate or potentially fraudulent.
4. Generate a fraud risk probability for individual transactions.
5. Provide a simple user interface for testing transactions.
6. Demonstrate the application of AI and machine learning to financial security.

## 4. Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Classifier
* Matplotlib
* Streamlit
* Joblib

## 5. Machine Learning Methodology

The project follows these major stages:

### Data Generation

A synthetic mobile money transaction dataset was created containing transaction and account characteristics.

### Data Preprocessing

Categorical transaction types were encoded using One-Hot Encoding, while numerical features were passed directly to the machine learning model.

### Feature Selection

The model uses:

* Transaction amount
* Transaction type
* Previous account balance
* Transaction frequency
* Account age
* Transaction hour
* Device change
* Location change

### Model Training

A Random Forest Classifier was trained using an 80/20 training and testing split.

The model uses class balancing to improve its ability to identify fraudulent transactions.

### Model Evaluation

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

## 6. Model Performance

The selected model achieved approximately:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 94.90% |
| Precision | 40.91% |
| Recall    | 85.14% |
| F1 Score  | 55.26% |

The recall score is particularly relevant to fraud detection because it indicates the model's ability to identify fraudulent transactions among actual fraudulent cases.

## 7. Application Features

The Streamlit application allows users to enter transaction information and receive:

* Fraud classification
* Fraud probability
* Risk level
* Transaction details
* Recommended action for suspicious transactions

## 8. System Workflow

The system follows this workflow:

User Transaction
↓
Data Input
↓
Data Preprocessing
↓
Machine Learning Model
↓
Fraud Probability
↓
Risk Classification
↓
Fraud Detection Result

## 9. Limitations

This project is an educational prototype and uses a synthetic dataset. Real-world mobile money fraud detection would require large amounts of validated transaction data and additional information such as device fingerprints, network information, historical customer behavior, merchant information, and confirmed fraud cases.

The model should therefore not be considered a production financial fraud detection system.

## 10. Future Improvements

Future versions could include:

* Real-world transaction datasets
* Real-time transaction monitoring
* Deep learning models
* Anomaly detection
* Customer behavioral profiling
* Explainable AI
* SMS and mobile notification alerts
* Integration with mobile money platforms
* Automated fraud investigation workflows

## 11. Conclusion

The Mobile Money Fraud Detector demonstrates how machine learning can be applied to financial security. By analyzing transaction and behavioral features, the system can identify suspicious transactions and provide a fraud risk assessment.

Although the current implementation is a prototype based on synthetic data, it provides a foundation for developing more advanced real-time mobile money fraud detection systems.
