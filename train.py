import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Media_analytics_synthetic_50k.csv")

df.drop_duplicates(inplace=True)
df.fillna(0, inplace=True)

# --------------------------------------------------
# CHURN MODEL (USER LEVEL)
# --------------------------------------------------

user_df = df.groupby('user_id').agg(
    watch_mean=('watch_duration_minutes', 'mean'),
    watch_sum=('watch_duration_minutes', 'sum'),
    ads_mean=('ad_exposure_count', 'mean'),
    ads_skipped_sum=('ad_skipped', 'sum'),
    buffering_sum=('buffering_events', 'sum'),
    rating_mean=('rating_given', 'mean')
).reset_index()

# Create probabilistic churn label
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

model_churn = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=0
)

model_churn.fit(X_train, y_train)
model_churn.save_model("churn_model.cbm")

print("Churn model trained and saved.")

# --------------------------------------------------
# AD EXPOSURE MODEL (LIMITED FEATURES ONLY)
# --------------------------------------------------

# Use only selected safe features
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

cat_features = X_reg.select_dtypes(include=['object']).columns.tolist()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

model_reg = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    verbose=0
)

model_reg.fit(X_train_r, y_train_r, cat_features=cat_features)
model_reg.save_model("ad_model.cbm")

print("Ad exposure model trained and saved.")

