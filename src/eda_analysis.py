import pandas as pd
import sqlite3

# Load the cleaned datasets
order_data = pd.read_csv("data/clean/orders.csv")
customer_data = pd.read_csv("data/clean/customers.csv")
product_data = pd.read_csv("data/clean/products.csv")

# Convert order timestamps into proper datetime format
order_data["order_ts"] = pd.to_datetime(order_data["order_ts"])

# Keep only completed orders for revenue analysis
completed_order_data = order_data[
    order_data["status"] == "completed"
].copy()

# 1. PANDAS CROSS-CHECK: REVENUE BY CITY AND CATEGORY
# Add customer city details to the completed orders
orders_with_city = completed_order_data.merge(
    customer_data[["customer_id", "city"]],
    on="customer_id"
)

# Add product category details to the orders
orders_with_city_category = orders_with_city.merge(
    product_data[["product_id", "category"]],
    on="product_id"
)

# Create a pivot table to compare revenue across cities and categories
revenue_pivot_table = pd.pivot_table(
    orders_with_city_category,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    fill_value=0
)
print("\n========== REVENUE PIVOT ==========")
print(revenue_pivot_table)

# 2. CITY-WISE REVENUE ANALYSIS
# Calculate revenue, average order value and completed orders for each city
city_revenue_analysis = orders_with_city.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    completed_orders=("order_id", "count")
)
print("\n========== CITY ANALYSIS ==========")
print(city_revenue_analysis)

# 3. COMPLETED ORDER AMOUNT STATISTICS
# Check the distribution of completed order amounts
print("\n========== ORDER AMOUNT STATISTICS ==========")
print(completed_order_data["order_amount"].describe())
print("\nMean:", completed_order_data["order_amount"].mean())
print("Median:", completed_order_data["order_amount"].median())
print("Skewness:", completed_order_data["order_amount"].skew())


# 4. OUTLIER DETECTION USING IQR
# Find the first and third quartiles of order amounts
first_quartile = completed_order_data["order_amount"].quantile(0.25)
third_quartile = completed_order_data["order_amount"].quantile(0.75)

# Calculate the interquartile range
interquartile_range = third_quartile - first_quartile

# Calculate the lower and upper limits for detecting outliers
lower_fence = first_quartile - 1.5 * interquartile_range
upper_fence = third_quartile + 1.5 * interquartile_range

# Identify orders whose amounts fall outside these limits
outlier_order_data = completed_order_data[
    (completed_order_data["order_amount"] < lower_fence) |
    (completed_order_data["order_amount"] > upper_fence)
]
print("\n========== OUTLIER ANALYSIS ==========")
print("Q1:", first_quartile)
print("Q3:", third_quartile)
print("IQR:", interquartile_range)
print("Lower Fence:", lower_fence)
print("Upper Fence:", upper_fence)
print("Outlier Count:", len(outlier_order_data))

# 5. CORRELATION ANALYSIS
# Check how order amount, discount, quantity and delivery time are related
correlation_matrix = completed_order_data[
    ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
].corr()
print("\n========== CORRELATION MATRIX ==========")
print(correlation_matrix)

# Count outliers separately below and above the calculated limits
lower_outlier_data = completed_order_data[
    completed_order_data["order_amount"] < lower_fence
]
upper_outlier_data = completed_order_data[
    completed_order_data["order_amount"] > upper_fence
]
print("Lower outliers:", len(lower_outlier_data))
print("Upper outliers:", len(upper_outlier_data))

# 6. COMPARE PANDAS REVENUE WITH SQL REVENUE
# Get the total revenue calculated by the Pandas pivot table
pandas_total_revenue = revenue_pivot_table.loc["All", "All"]
print("\nPandas grand total:", pandas_total_revenue)

# Connect to the Instamart SQLite database
database_connection = sqlite3.connect("data/instamart.db")

# Calculate completed order revenue directly using SQL
sql_revenue_result = pd.read_sql_query(
    """
    SELECT SUM(order_amount) AS total_revenue
    FROM orders
    WHERE status = 'completed'
    """,
    database_connection
)
sql_total_revenue = sql_revenue_result["total_revenue"].iloc[0]

# Close the database connection after the query
database_connection.close()
print("SQL revenue:", sql_total_revenue)

# Make sure Pandas and SQL return the same total revenue
assert round(pandas_total_revenue, 2) == round(sql_total_revenue, 2)

# 7. IDENTIFY THE STRONGEST CORRELATION
# Take absolute values to compare correlation strength
absolute_correlation = correlation_matrix.abs().copy()

# Remove diagonal values so a variable is not compared with itself
diagonal_mask = pd.DataFrame(
    absolute_correlation.index.to_numpy()[:, None] ==
    absolute_correlation.columns.to_numpy(),
    index=absolute_correlation.index,
    columns=absolute_correlation.columns
)
absolute_correlation = absolute_correlation.where(~diagonal_mask)

# Find the variable pair with the strongest correlation
strongest_correlation_pair = absolute_correlation.unstack().idxmax()
print("Strongest correlation pair:", strongest_correlation_pair)

# Display the original correlation value for that pair
print(
    "Correlation value:",
    correlation_matrix.loc[
        strongest_correlation_pair[0],
        strongest_correlation_pair[1]
    ]
)