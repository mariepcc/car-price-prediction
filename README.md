## Used Car Price Prediction

Machine learning project predicting used car prices based on technical parameters and vehicle history.
Built with scikit-learn and TensorFlow/Keras.

**Dataset:** [Used Car Price Prediction — Kaggle](https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset)  
~4,000 records · USA used car listings · features: mileage, engine, transmission, accident history

---

## Models & Results

| Model | R² | RMSE | MAE |
|:---|:---:|:---:|:---:|
| Random Forest | 0.857 | $8,739 | $5,926 |
| MLP (sklearn) | 0.832 | $9,833 | $6,805 |
| SVR | 0.824 | $10,130 | $6,738 |
| MLP (Keras) | 0.807 | $10,822 | $7,379 |

---

## Pipeline

1. Data cleaning — parsing `price`, `milage`, extracting `hp`, `engine_size`, `cylinders` from raw text
2. Feature engineering — `car_age`, log transformation of target
3. Outlier removal — IQR method
4. Preprocessing — LabelEncoder + StandardScaler
5. Hyperparameter tuning — RandomizedSearchCV (RF, SVR, MLP) · architecture search (Keras)
6. Evaluation — RMSE, MAE, R² · Permutation Importance for all models
