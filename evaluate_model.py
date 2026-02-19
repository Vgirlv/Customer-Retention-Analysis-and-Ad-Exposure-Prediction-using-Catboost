import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    mean_squared_error,
    r2_score
)
from sklearn.preprocessing import MinMaxScaler

plt.switch_backend('TkAgg')  # ensures plots open in VS Code

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Media_analytics_synthetic_50k.csv")
df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# ==================================================
# CHURN MODEL EVALUATION
# ==================================================

print("\n============================")
print("   CHURN MODEL EVALUATION")
print("============================")

# Recreate user-level data
user_df = df.groupby('user_id').agg(
    watch_mean=('watch_duration_minutes', 'mean'),
    watch_sum=('watch_duration_minutes', 'sum'),
    ads_mean=('ad_exposure_count', 'mean'),
    ads_skipped_sum=('ad_skipped', 'sum'),
    buffering_sum=('buffering_events', 'sum'),
    rating_mean=('rating_given', 'mean')
).reset_index()

# Recreate churn label
scaler = MinMaxScaler()
scaled = scaler.fit_transform(
    user_df[['watch_mean','buffering_sum','ads_skipped_sum','rating_mean']]
)

scaled_df = pd.DataFrame(scaled, columns=['watch','buffer','ads','rating'])

churn_prob = (
    0.5*(1 - scaled_df['watch']) +
    0.2*scaled_df['buffer'] +
    0.2*scaled_df['ads'] +
    0.1*(1 - scaled_df['rating'])
)

noise = np.random.normal(0, 0.1, size=len(churn_prob))
churn_prob = churn_prob + noise

user_df['churn'] = (churn_prob > churn_prob.median()).astype(int)

X = user_df.drop(columns=['user_id','churn'])
y = user_df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Load saved churn model
model_churn = CatBoostClassifier()
model_churn.load_model("churn_model.cbm")

y_pred = model_churn.predict(X_test)
y_prob = model_churn.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Churn")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve - Churn")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

# Feature Importance
importance = model_churn.get_feature_importance()
features = X.columns

plt.figure()
plt.barh(features, importance)
plt.title("Feature Importance - Churn")
plt.show()

# ==================================================
# AD EXPOSURE MODEL EVALUATION
# ==================================================

print("\n============================")
print(" AD EXPOSURE MODEL EVALUATION")
print("============================")

ad_features = [
    'subscription_type',
    'watch_duration_minutes',
    'ad_skipped',
    'device_type',
    'network_type',
    'buffering_events'
]

X_reg = df[ad_features]
y_reg = df['ad_exposure_count']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# Load saved regression model
model_reg = CatBoostRegressor()
model_reg.load_model("ad_model.cbm")

y_pred_reg = model_reg.predict(X_test_r)

print("MSE:", mean_squared_error(y_test_r, y_pred_reg))
print("R2 Score:", r2_score(y_test_r, y_pred_reg))

# Feature Importance (Regression)
importance_reg = model_reg.get_feature_importance()
features_reg = ad_features

plt.figure()
plt.barh(features_reg, importance_reg)
plt.title("Feature Importance - Ad Exposure")
plt.show()

print("\nEvaluation Complete.")
