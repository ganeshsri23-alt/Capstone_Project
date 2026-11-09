import pandas as pd
import os

# TASK 2: CHECK THE RAW DATA
# Load all three datasets from the raw data folder
data_path = "data/raw/"
raw_customers = pd.read_csv(data_path + "customers.csv")
raw_products = pd.read_csv(data_path + "products.csv")
raw_orders = pd.read_csv(data_path + "orders.csv")

# Keep all datasets together so we can check them one by one
raw_datasets = {
    "Customers": raw_customers,
    "Products": raw_products,
    "Orders": raw_orders
}
# Check the shape, data types and missing values in each dataset
for table_name, table_data in raw_datasets.items():
    print("\n==========", table_name, "==========")
    print("Shape:", table_data.shape)
    print("\nData Types:")
    print(table_data.dtypes)

    missing_values = table_data.isnull().sum()
    missing_percent = (missing_values / len(table_data)) * 100
    missing_summary = pd.DataFrame({
        "Missing Count": missing_values,
        "Missing Percentage": missing_percent
    })

    print("\nMissing Values:")
    print(missing_summary)

# Check important data quality issues in orders
print("\nExact duplicate rows:", raw_orders.duplicated().sum())
print("Pending orders:", (raw_orders["status"] == "pending").sum())
print("Distinct order IDs:", raw_orders["order_id"].nunique())
# Convert order amounts into numbers to identify invalid or blank values
numeric_amounts = pd.to_numeric(
    raw_orders["order_amount"], errors="coerce"
)
print("Non-numeric or blank order amounts:", numeric_amounts.isna().sum())
print("Negative delivery minutes:", (raw_orders["delivery_minutes"] < 0).sum())
print("Blank delivery minutes:", raw_orders["delivery_minutes"].isna().sum())
# Check customer age, city and gender values
print("\nBlank ages:", raw_customers["age"].isna().sum())

invalid_ages = (
    (raw_customers["age"] < 10) |
    (raw_customers["age"] > 100)
)
print("Ages outside 10-100:", invalid_ages.sum())
print("Distinct raw city spellings:", raw_customers["city"].nunique())
print(
    "Distinct non-blank raw gender spellings:",
    raw_customers["gender"].dropna().nunique()
)
print("Blank genders:", raw_customers["gender"].isna().sum())
# TASK 3: CLEAN CUSTOMER DATA
# Take a look at the customer data before making any changes
print(raw_customers.head(10))
raw_customers.info()
print("Duplicate rows:", raw_customers.duplicated().sum())
print("Duplicate customer IDs:", raw_customers["customer_id"].duplicated().sum())

# Check how customer ages are distributed
print(raw_customers["age"].describe())
invalid_age_rows = raw_customers[
    (raw_customers["age"] < 10) |
    (raw_customers["age"] > 100)
]
print("Invalid age count:", len(invalid_age_rows))
print(invalid_age_rows[["customer_id", "age"]])

# Create a separate copy so the raw customer data stays unchanged
clean_customers = raw_customers.copy()
# Remove extra spaces and fix city name capitalization
clean_customers["city"] = (
    clean_customers["city"]
    .str.strip()
    .str.title()
)
city_changes = (
    raw_customers["city"] != clean_customers["city"]
).sum()
print("\nCity values standardized:", city_changes)
print("\nCities after standardization:")
print(clean_customers["city"].value_counts())
print("\nNumber of unique cities:", clean_customers["city"].nunique())

# Convert different gender spellings into a common format
gender_map = {
    "female": "female",
    "f": "female",
    "male": "male",
    "m": "male"
}
clean_customers["gender"] = (
    clean_customers["gender"]
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("unknown")
)
blank_gender_count = raw_customers["gender"].isna().sum()
gender_changes = (
    raw_customers["gender"].notna() &
    (raw_customers["gender"] != clean_customers["gender"])
).sum()
print("Gender values standardized (non-blank):", gender_changes)
print("Blank genders mapped to unknown:", blank_gender_count)
print("\nGender counts:")
print(clean_customers["gender"].value_counts())

# Treat ages below 10 or above 100 as invalid values
original_blank_ages = clean_customers["age"].isna().sum()
invalid_age_count = (
    (clean_customers["age"] < 10) |
    (clean_customers["age"] > 100)
).sum()
print("\nOriginal blank ages:", original_blank_ages)
print("Age outliers masked:", invalid_age_count)

# Replace invalid ages with missing values before filling them
clean_customers.loc[
    (clean_customers["age"] < 10) |
    (clean_customers["age"] > 100),
    "age"
] = pd.NA
missing_ages_before = clean_customers["age"].isna().sum()
print("Missing ages before median imputation:", missing_ages_before)

# Fill missing ages using the median age of the same city
clean_customers["age"] = (
    clean_customers["age"].fillna(
        clean_customers.groupby("city")["age"].transform("median")
    )
)
missing_ages_after = clean_customers["age"].isna().sum()
print(
    "Ages filled using city median:",
    missing_ages_before - missing_ages_after
)
print("Missing ages after median imputation:", missing_ages_after)

# Convert signup dates into proper datetime format
clean_customers["signup_date"] = pd.to_datetime(
    clean_customers["signup_date"]
)
# Check the final age values after cleaning
print("\nFinal Age Summary:")
print(clean_customers["age"].describe())
print("\nMinimum age:", clean_customers["age"].min())
print("Maximum age:", clean_customers["age"].max())

# TASK 4: CLEAN ORDER DATA
# Work on a copy of the original orders
clean_orders = raw_orders.copy()

# Convert order timestamps into proper datetime format
clean_orders["order_ts"] = pd.to_datetime(
    clean_orders["order_ts"]
)
# Remove rows that are completely duplicated
rows_before_cleaning = len(clean_orders)
clean_orders = clean_orders.drop_duplicates()
print(
    "\nExact duplicate order rows removed:",
    rows_before_cleaning - len(clean_orders)
)

# Keep only the latest record for each order ID
# Stable sorting preserves the original order when timestamps are equal
rows_before_cleaning = len(clean_orders)
clean_orders = (
    clean_orders
    .sort_values("order_ts", kind="stable")
    .drop_duplicates(subset="order_id", keep="last")
)
print(
    "Older order_id records removed:",
    rows_before_cleaning - len(clean_orders)
)
# Convert order amounts into numeric values
clean_orders["order_amount"] = pd.to_numeric(
    clean_orders["order_amount"], errors="coerce"
)
# Bring product price and category into the orders data
clean_orders = clean_orders.merge(
    raw_products[["product_id", "category", "price_inr"]],
    on="product_id",
    how="left"
)
# Fill missing order amounts using product price, quantity and discount
missing_amounts = clean_orders["order_amount"].isnull()
print("Missing order amounts to fill:", missing_amounts.sum())
clean_orders.loc[missing_amounts, "order_amount"] = (
    clean_orders.loc[missing_amounts, "price_inr"]
    * clean_orders.loc[missing_amounts, "quantity"]
    * (
        1 -
        clean_orders.loc[missing_amounts, "discount_pct"] / 100
    )
).round(2)
# Negative delivery times are invalid, so replace them with missing values
negative_delivery_count = (
    clean_orders["delivery_minutes"] < 0
).sum()
print("Negative delivery minutes masked:", negative_delivery_count)
clean_orders.loc[
    clean_orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA
# Fill missing delivery times only for completed orders
completed_order_rows = clean_orders["status"] == "completed"
completed_missing_before = clean_orders.loc[
    completed_order_rows, "delivery_minutes"
].isna().sum()
# Use the median delivery time of the same product category
clean_orders.loc[completed_order_rows, "delivery_minutes"] = (
    clean_orders.loc[completed_order_rows, "delivery_minutes"]
    .fillna(
        clean_orders.loc[completed_order_rows]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )
)
completed_missing_after = clean_orders.loc[
    completed_order_rows, "delivery_minutes"
].isna().sum()
print(
    "Completed delivery times filled:",
    completed_missing_before - completed_missing_after
)
# Cancelled orders should not have delivery times
cancelled_order_rows = clean_orders["status"] == "cancelled"
print(
    "Cancelled delivery values cleared:",
    clean_orders.loc[
        cancelled_order_rows, "delivery_minutes"
    ].notna().sum()
)
clean_orders.loc[
    cancelled_order_rows, "delivery_minutes"
] = pd.NA
# Remove product details that were temporarily added for cleaning
clean_orders = clean_orders.drop(
    columns=["category", "price_inr"]
)
# Check whether the cleaned orders meet the expected conditions
print("\nCleaned orders shape:", clean_orders.shape)
print("\nOrder status counts:")
print(clean_orders["status"].value_counts())
print("\nUnique order IDs:", clean_orders["order_id"].nunique())
print(
    "\nMissing order amounts:",
    clean_orders["order_amount"].isnull().sum()
)
print(
    "\nMissing delivery minutes in completed orders:",
    clean_orders.loc[
        clean_orders["status"] == "completed",
        "delivery_minutes"
    ].isnull().sum()
)
print(
    "\nMissing delivery minutes in cancelled orders:",
    clean_orders.loc[
        clean_orders["status"] == "cancelled",
        "delivery_minutes"
    ].isnull().sum()
)
print(
    "\nCompleted order amount total INR:",
    clean_orders.loc[
        clean_orders["status"] == "completed",
        "order_amount"
    ].sum().round(2)
)
# SAVE THE CLEANED DATASETS
# Create the clean folder if it does not already exist
os.makedirs("data/clean", exist_ok=True)

# Save all cleaned datasets as CSV files
clean_customers.to_csv(
    "data/clean/customers.csv",
    index=False
)
print("Cleaned customers dataset saved successfully")
clean_orders.to_csv(
    "data/clean/orders.csv",
    index=False
)
print("Cleaned orders dataset saved successfully")
raw_products.to_csv(
    "data/clean/products.csv",
    index=False
)
print("Cleaned products dataset saved successfully")

# CHECK THE SAVED FILES
# Read the saved CSV files and verify their shapes
saved_customers = pd.read_csv("data/clean/customers.csv")
saved_products = pd.read_csv("data/clean/products.csv")
saved_orders = pd.read_csv("data/clean/orders.csv")
print("\nCustomers shape:", saved_customers.shape)
print("Products shape:", saved_products.shape)
print("Orders shape:", saved_orders.shape)