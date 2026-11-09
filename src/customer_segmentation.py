import pandas as pd
import numpy as np

# Load cleaned data
order_data = pd.read_csv("data/clean/orders.csv")
customer_data = pd.read_csv("data/clean/customers.csv")

# Convert order timestamp
order_data["order_ts"] = pd.to_datetime(order_data["order_ts"])

# Snapshot date
snapshot_date = pd.Timestamp("2025-10-01 00:00:00")

# Select completed orders
completed_order_data = order_data[
    order_data["status"] == "completed"
].copy()

# Build RFM table
rfm_data = completed_order_data.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

# Calculate recency
rfm_data["recency"] = (
    snapshot_date - rfm_data["last_order_ts"]
).dt.days

# Remove last order timestamp
rfm_data = rfm_data[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Display RFM results
print("\n========== RFM TABLE ==========")
print(rfm_data.head())
print("\n========== RFM SUMMARY ==========")
print(rfm_data[["recency", "frequency", "monetary"]].describe())
print("\nCustomer count:", len(rfm_data))
print("Recency median:", rfm_data["recency"].median())
print("Recency minimum:", rfm_data["recency"].min())
print("Recency maximum:", rfm_data["recency"].max())
print("Frequency mean:", rfm_data["frequency"].mean())
print("Frequency maximum:", rfm_data["frequency"].max())
print("Monetary mean:", rfm_data["monetary"].mean())
print("Monetary maximum:", rfm_data["monetary"].max())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Select RFM features
rfm_features = rfm_data[
    ["recency", "frequency", "monetary"]
]

# Standardise RFM features
feature_scaler = StandardScaler()
scaled_rfm_features = feature_scaler.fit_transform(rfm_features)

# Store clustering results
clustering_results = []

# Test K values from 2 to 8
for k in range(2, 9):
    kmeans_model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )
    cluster_labels = kmeans_model.fit_predict(scaled_rfm_features)
    cluster_inertia = kmeans_model.inertia_
    silhouette_score_value = silhouette_score(
        scaled_rfm_features,
        cluster_labels
    )
    clustering_results.append({
        "K": k,
        "Inertia": cluster_inertia,
        "Silhouette": silhouette_score_value
    })

# Display results
clustering_results_df = pd.DataFrame(clustering_results)
print("\n========== K-MEANS RESULTS ==========")
print(clustering_results_df.round(3))

from sklearn.decomposition import PCA
import os

# TASK 3: FINAL CLUSTERING WITH K = 4
final_kmeans_model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm_data["segment_id"] = final_kmeans_model.fit_predict(scaled_rfm_features)

# Profile each segment
segment_profile_data = rfm_data.groupby("segment_id").agg(
    customer_count=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean")
)
print("\n========== SEGMENT PROFILE ==========")
print(segment_profile_data.round(2))

# PCA: REDUCE RFM TO TWO DIMENSIONS

pca_model = PCA(n_components=2)
rfm_pca_data = pca_model.fit_transform(scaled_rfm_features)
print("\n========== PCA RESULTS ==========")
print("Explained variance ratio:", pca_model.explained_variance_ratio_)
print("Total explained variance:", pca_model.explained_variance_ratio_.sum())

# NAME SEGMENTS
# Identify premium segment
premium_segment_id = segment_profile_data["mean_monetary"].idxmax()

# Assign business names based on segment profiles
segment_name_mapping = {}
for segment_id, profile in segment_profile_data.iterrows():
    if segment_id == premium_segment_id:
        segment_name_mapping[segment_id] = "Premium Customers"
    elif profile["mean_recency"] > rfm_data["recency"].mean():
        segment_name_mapping[segment_id] = "Inactive Customers"
    elif profile["mean_frequency"] > rfm_data["frequency"].mean():
        segment_name_mapping[segment_id] = "Frequent Customers"
    else:
        segment_name_mapping[segment_id] = "Occasional Customers"

# Assign segment names
rfm_data["segment_name"] = rfm_data["segment_id"].map(segment_name_mapping)

# Flag premium customers
rfm_data["is_premium"] = (
    rfm_data["segment_id"] == premium_segment_id
)
print("\nPremium segment ID:", premium_segment_id)

# SAVE CUSTOMER SEGMENTS
os.makedirs("outputs", exist_ok=True)
customer_segments = rfm_data[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]
customer_segments.to_csv(
    "outputs/customer_segments.csv",
    index=False
)
print("\nCustomer segments saved successfully.")
print(customer_segments.head())

# TASK 4: LAPSE PREDICTION DATASET
# Define cutoff date and lapse observation period
cutoff = pd.Timestamp("2025-09-01")
observation_end = pd.Timestamp("2025-10-01")

# Completed orders before cutoff
pre_cutoff_order_data = completed_order_data[
    completed_order_data["order_ts"] < cutoff
].copy()

# Customers who purchased before cutoff
lapse_customer_cohort = pre_cutoff_order_data["customer_id"].unique()

# Customers who purchased during September
september_order_data = completed_order_data[
    (completed_order_data["order_ts"] >= cutoff) &
    (completed_order_data["order_ts"] < observation_end)
]
september_customer_ids = september_order_data["customer_id"].unique()

# Create cohort dataset
lapse_label_data = pd.DataFrame({
    "customer_id": lapse_customer_cohort
})

# Label customers who did not purchase during September
lapse_label_data["lapsed"] = (
    ~lapse_label_data["customer_id"].isin(september_customer_ids)
).astype(int)

# Display results
print("\n========== LAPSE COHORT RESULTS ==========")
print("Cohort size:", len(lapse_label_data))
print("Lapsed customers:", lapse_label_data["lapsed"].sum())
print("Lapse percentage:", round(lapse_label_data["lapsed"].mean() * 100, 2))
print("\nTarget distribution:")
print(lapse_label_data["lapsed"].value_counts())

# TASK 4: CUSTOMER FEATURES
# Recency, frequency and monetary before cutoff
customer_feature_data = pre_cutoff_order_data.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean")
).reset_index()

# Recency calculated from cutoff date
customer_feature_data["recency"] = (
    cutoff - customer_feature_data["last_order"]
).dt.days

# Cancellation rate before cutoff
all_pre_cutoff_order_data = order_data[order_data["order_ts"] < cutoff].copy()
cancellation_rate_data = all_pre_cutoff_order_data.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled_orders=("status", lambda x: (x == "cancelled").sum())
).reset_index()
cancellation_rate_data["cancellation_rate"] = (
    cancellation_rate_data["cancelled_orders"] /
    cancellation_rate_data["total_orders"]
)

# Merge cancellation rate
customer_feature_data = customer_feature_data.merge(
    cancellation_rate_data[["customer_id", "cancellation_rate"]],
    on="customer_id",
    how="left"
)

# Customer details
customer_feature_data = customer_feature_data.merge(
    customer_data[["customer_id", "city", "age", "gender", "signup_date"]],
    on="customer_id",
    how="left"
)

# Customer tenure as of cutoff
customer_feature_data["signup_date"] = pd.to_datetime(
    customer_feature_data["signup_date"]
)
customer_feature_data["tenure_days"] = (
    cutoff - customer_feature_data["signup_date"]
).dt.days

# Load products
product_data = pd.read_csv("data/clean/products.csv")

# Add product category to pre-cutoff orders
pre_cutoff_order_data = pre_cutoff_order_data.merge(
    product_data[["product_id", "category"]],
    on="product_id",
    how="left"
)

# Find most frequently purchased category for each customer
favorite_category_data = pre_cutoff_order_data.groupby("customer_id")[
    "category"
].agg(lambda x: x.mode().iloc[0]).reset_index()
favorite_category_data.columns = ["customer_id", "favorite_category"]

# Merge customer features and favorite category
customer_feature_data = customer_feature_data.merge(
    favorite_category_data,
    on="customer_id",
    how="left"
)

# Merge target label
lapse_prediction_data = customer_feature_data.merge(
    lapse_label_data[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner"
)
print("\nLAPSE DATASET COLUMNS:")
print(lapse_prediction_data.columns.tolist())

# TASK 4: TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split

# Define features and target
feature_data = lapse_prediction_data.drop(
    columns=["customer_id", "last_order", "signup_date", "lapsed"]
)
target_data = lapse_prediction_data["lapsed"]

# Split dataset
training_features, testing_features, training_target, testing_target = train_test_split(
    feature_data,
    target_data,
    test_size=0.20,
    random_state=42,
    stratify=target_data
)
print("\n========== TRAIN-TEST SPLIT ==========")
print("Training rows:", len(training_features))
print("Testing rows:", len(testing_features))
print("\nTraining target distribution:")
print(training_target.value_counts())
print("\nTesting target distribution:")
print(testing_target.value_counts())

# TASK 5: PREPROCESSING
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Numerical and categorical columns
numerical_feature_columns = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "tenure_days",
    "age"
]
categorical_feature_columns = [
    "city",
    "gender",
    "favorite_category"
]

# Preprocessing
data_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_feature_columns),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
         categorical_feature_columns)
    ]
)

# Fit only on training data
processed_training_features = data_preprocessor.fit_transform(training_features)

# Transform test data without fitting
processed_testing_features = data_preprocessor.transform(testing_features)
print("\n========== PREPROCESSING RESULTS ==========")
print("Training shape:", processed_training_features.shape)
print("Testing shape:", processed_testing_features.shape)
print("\nMissing values in X_train:", training_features.isnull().sum().sum())
print("Missing values in X_test:", testing_features.isnull().sum().sum())

# Compare scaler training mean with full-cohort mean
training_frequency_mean = data_preprocessor.named_transformers_["num"].mean_[1]
cohort_frequency_mean = lapse_prediction_data["frequency"].mean()
print("\nScaler frequency mean:", training_frequency_mean)
print("Full cohort frequency mean:", cohort_frequency_mean)

# TASK 6: LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Logistic Regression pipeline
logistic_regression_pipeline = Pipeline([
    ("preprocessor", data_preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

# Hyperparameter grid
logistic_regression_parameter_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

# Grid Search
logistic_regression_grid_search = GridSearchCV(
    logistic_regression_pipeline,
    logistic_regression_parameter_grid,
    cv=5,
    scoring="f1"
)
logistic_regression_grid_search.fit(training_features, training_target)
print("\n========== LOGISTIC REGRESSION ==========")
print("Best parameters:", logistic_regression_grid_search.best_params_)
print("Best CV F1:", logistic_regression_grid_search.best_score_)

# TASK 6: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
random_forest_pipeline = Pipeline([
    ("preprocessor", data_preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])
random_forest_parameter_grid = {
    "model__max_depth": [3, 5, 8],
    "model__min_samples_leaf": [5, 10, 20]
}
random_forest_grid_search = GridSearchCV(
    random_forest_pipeline,
    random_forest_parameter_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
random_forest_grid_search.fit(training_features, training_target)
print("\n========== RANDOM FOREST ==========")
print("Best parameters:", random_forest_grid_search.best_params_)
print("Best CV F1:", random_forest_grid_search.best_score_)

# TASK 6: MODEL EVALUATION
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

# Best models
best_logistic_regression_model = logistic_regression_grid_search.best_estimator_
best_random_forest_model = random_forest_grid_search.best_estimator_

# Predictions on untouched test data
logistic_regression_predictions = best_logistic_regression_model.predict(testing_features)
random_forest_predictions = best_random_forest_model.predict(testing_features)

# Baseline model: predict every customer as not lapsed (0)
baseline_model_predictions = np.zeros(len(testing_target), dtype=int)
baseline_model_accuracy = accuracy_score(testing_target, baseline_model_predictions)
baseline_model_f1 = f1_score(testing_target, baseline_model_predictions, zero_division=0)
print("\n========== BASELINE MODEL ==========")
print("Baseline Accuracy:", baseline_model_accuracy)
print("Baseline Lapsed-Class F1:", baseline_model_f1)

# Logistic Regression results
print("\n========== LOGISTIC REGRESSION TEST ==========")
print("Accuracy:", accuracy_score(testing_target, logistic_regression_predictions))
print("Confusion Matrix:")
print(confusion_matrix(testing_target, logistic_regression_predictions))
print(classification_report(testing_target, logistic_regression_predictions, zero_division=0))

# Random Forest results
print("\n========== RANDOM FOREST TEST ==========")
print("Accuracy:", accuracy_score(testing_target, random_forest_predictions))
print("Confusion Matrix:")
print(confusion_matrix(testing_target, random_forest_predictions))
print(classification_report(testing_target, random_forest_predictions, zero_division=0))

# Compare baseline and tuned models
print("\n========== MODEL COMPARISON ==========")
print("Baseline Accuracy:", baseline_model_accuracy)
print("Baseline F1:", baseline_model_f1)
print("Logistic Regression F1:", f1_score(testing_target, logistic_regression_predictions))
print("Random Forest F1:", f1_score(testing_target, random_forest_predictions))

# TASK 7: OVERSAMPLING
from sklearn.utils import resample
from sklearn.metrics import precision_score, recall_score, f1_score

# Combine training features and target
training_data = training_features.copy()
training_data["lapsed"] = training_target.values

# Separate majority and minority classes
majority_class_data = training_data[training_data["lapsed"] == 0]
minority_class_data = training_data[training_data["lapsed"] == 1]

# Calculate how many extra lapsed rows are needed
required_extra_samples = len(majority_class_data) - len(minority_class_data)

# Randomly duplicate only the required number of lapsed rows
oversampled_minority_data = resample(
    minority_class_data,
    replace=True,
    n_samples=required_extra_samples,
    random_state=42
)

# Keep all original rows and add the extra lapsed rows
balanced_training_data = pd.concat(
    [majority_class_data, minority_class_data, oversampled_minority_data]
)
balanced_training_features = balanced_training_data.drop(columns=["lapsed"])
balanced_training_target = balanced_training_data["lapsed"]
print("\n========== OVERSAMPLING RESULTS ==========")
print(balanced_training_target.value_counts())

# Train Logistic Regression on balanced data
oversampled_logistic_regression_model = Pipeline([
    ("preprocessor", data_preprocessor),
    ("model", LogisticRegression(
        C=logistic_regression_grid_search.best_params_["model__C"],
        max_iter=1000
    ))
])
oversampled_logistic_regression_model.fit(balanced_training_features, balanced_training_target)

# Evaluate on original test data
oversampled_model_predictions = oversampled_logistic_regression_model.predict(testing_features)
print("\n========== OVERSAMPLING COMPARISON ==========")
print("\nOriginal Tuned Logistic Regression:")
print("Precision:", precision_score(testing_target, logistic_regression_predictions))
print("Recall:", recall_score(testing_target, logistic_regression_predictions))
print("F1:", f1_score(testing_target, logistic_regression_predictions))
print("\nOversampled Logistic Regression:")
print("Precision:", precision_score(testing_target, oversampled_model_predictions))
print("Recall:", recall_score(testing_target, oversampled_model_predictions))
print("F1:", f1_score(testing_target, oversampled_model_predictions))

# TASK 8: FEATURE IMPORTANCE
# Random Forest feature importance
random_forest_model = best_random_forest_model.named_steps["model"]
processed_feature_names = best_random_forest_model.named_steps[
    "preprocessor"
].get_feature_names_out()
random_forest_feature_importance = pd.DataFrame({
    "feature": processed_feature_names,
    "importance": random_forest_model.feature_importances_
}).sort_values("importance", ascending=False)
print("\n========== TOP 5 RF FEATURES ==========")
print(random_forest_feature_importance.head(5))

# Logistic Regression coefficients
logistic_regression_model = best_logistic_regression_model.named_steps["model"]
logistic_regression_feature_names = best_logistic_regression_model.named_steps[
    "preprocessor"
].get_feature_names_out()
logistic_regression_coefficients = pd.DataFrame({
    "feature": logistic_regression_feature_names,
    "coefficient": logistic_regression_model.coef_[0]
})

# Select top 5 Random Forest features
top_five_features = random_forest_feature_importance.head(5).copy()

# Match RF features with LR coefficients
top_five_features = top_five_features.merge(
    logistic_regression_coefficients,
    on="feature",
    how="left",
    validate="one_to_one"
)

# Identify LR coefficient direction
top_five_features["lr_sign"] = np.where(
    top_five_features["coefficient"] > 0,
    "Positive",
    np.where(
        top_five_features["coefficient"] < 0,
        "Negative",
        "Zero"
    )
)

# Feature-specific business descriptions
feature_business_meanings = {
    "recency": "Days since last order",
    "frequency": "Number of orders",
    "monetary": "Total customer spending",
    "avg_discount": "Average discount received",
    "cancellation_rate": "Proportion of cancelled orders",
    "avg_delivery": "Average delivery time",
    "tenure_days": "Days since signup",
    "age": "Customer age"
}

# Generate one business interpretation per feature
def explain_feature_business_meaning(row):
    feature = row["feature"].split("__")[-1]
    meaning = feature_business_meanings.get(feature, feature)
    if pd.isna(row["coefficient"]):
        return f"{meaning}: LR coefficient unavailable."
    if row["coefficient"] > 0:
        direction = "higher predicted lapse risk"
    elif row["coefficient"] < 0:
        direction = "lower predicted lapse risk"
    else:
        direction = "no directional association with lapse risk"
    return (
        f"{meaning}: a higher value is associated with "
        f"{direction}."
    )
top_five_features["business_reading"] = top_five_features.apply(
    explain_feature_business_meaning,
    axis=1
)

# Display final feature importance comparison
print("\n========== TOP 5 RF FEATURES WITH LR SIGNS ==========")
print(
    top_five_features[
        [
            "feature",
            "importance",
            "coefficient",
            "lr_sign",
            "business_reading"
        ]
    ].to_string(index=False)
)

# TASK 8: SAVE AND RELOAD MODEL
import joblib
import os
os.makedirs("outputs", exist_ok=True)

# Select the final model: the tuned model with the higher lapsed-class F1 on the test set
logistic_regression_test_f1 = f1_score(testing_target, logistic_regression_predictions)
random_forest_test_f1 = f1_score(testing_target, random_forest_predictions)
final_selected_model = best_logistic_regression_model if logistic_regression_test_f1 >= random_forest_test_f1 else best_random_forest_model
final_selected_model_name = "Tuned Logistic Regression" if final_selected_model is best_logistic_regression_model else "Tuned Random Forest"

# Generate predictions before saving
predictions_before_saving = final_selected_model.predict(testing_features)

# Save the selected model
joblib.dump(final_selected_model, "outputs/lapse_model.joblib")

# Reload the saved model
reloaded_model = joblib.load("outputs/lapse_model.joblib")

# Generate predictions after reloading
predictions_after_reloading = reloaded_model.predict(testing_features)

# Verify predictions are identical
model_predictions_match = np.array_equal(
    predictions_before_saving,
    predictions_after_reloading
)
print("\n========== MODEL RELOAD VERIFICATION ==========")
print("Selected Model:", final_selected_model_name)
print("Predictions identical:", model_predictions_match)
assert model_predictions_match, "Model predictions changed after reloading!"

# TASK 8: RETENTION LIST
# Predict lapse probability for test customers
predicted_lapse_probabilities = reloaded_model.predict_proba(testing_features)[:, 1]

# Create test customer results
premium_retention_list = pd.DataFrame({
    "customer_id": lapse_prediction_data.loc[testing_features.index, "customer_id"].values,
    "lapse_probability": predicted_lapse_probabilities,
    "actually_lapsed": testing_target.values
})

# Load premium segment customers
customer_segments = pd.read_csv("outputs/customer_segments.csv")
premium_customer_ids = customer_segments[
    customer_segments["is_premium"] == True
]["customer_id"]

# Filter premium customers from test set
premium_retention_list = premium_retention_list[
    premium_retention_list["customer_id"].isin(premium_customer_ids)
].copy()

# Rank premium customers by highest lapse probability
premium_retention_list = premium_retention_list.sort_values(
    "lapse_probability",
    ascending=False
)

# Select only the top 10 premium customers
premium_retention_list = premium_retention_list.head(10).copy()

# Save the top 10 retention list
premium_retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)
print("\n========== TOP 10 PREMIUM RETENTION LIST ==========")
print("Customers selected:", len(premium_retention_list))
print("Actually lapsed:", premium_retention_list["actually_lapsed"].sum())
print("\nTop 10 Premium Customers:")
print(premium_retention_list.to_string(index=False))
print("\nTop 10 retention customers:")
print(premium_retention_list.head(10))