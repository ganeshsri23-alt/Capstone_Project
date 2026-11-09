# TASK 5: STORE AND VALIDATE
# Import the libraries needed for data handling and database storage
import pandas as pd
import sqlite3
import os

# STEP 1: CREATE REQUIRED DIRECTORIES
# Create the clean data folder and data folder if they do not exist
os.makedirs("data/clean", exist_ok=True)
os.makedirs("data", exist_ok=True)

# STEP 2: LOAD CLEANED DATASETS
# Load the cleaned customer, product, and order data from CSV files
customer_data = pd.read_csv("data/clean/customers.csv")
product_data = pd.read_csv("data/clean/products.csv")
order_data = pd.read_csv("data/clean/orders.csv")

# STEP 3: FORMAT ORDER TIMESTAMPS
# Convert order timestamps into the required date and time format
order_data["order_ts"] = pd.to_datetime(
    order_data["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

# STEP 4: SAVE CLEANED DATASETS AS CSV
# Save the cleaned datasets back into their CSV files
customer_data.to_csv("data/clean/customers.csv", index=False)
product_data.to_csv("data/clean/products.csv", index=False)
order_data.to_csv("data/clean/orders.csv", index=False)

# STEP 5: CREATE SQLITE DATABASE
# Connect to the Instamart database
database_connection = sqlite3.connect("data/instamart.db")

# STEP 6: STORE DATASETS AS SQLITE TABLES
# Store customer data in the customers table
customer_data.to_sql(
    "customers",
    database_connection,
    if_exists="replace",
    index=False
)

# Store product data in the products table
product_data.to_sql(
    "products",
    database_connection,
    if_exists="replace",
    index=False
)

# Store order data in the orders table
order_data.to_sql(
    "orders",
    database_connection,
    if_exists="replace",
    index=False
)

# VALIDATION CHECKPOINTS
# Check 1: Verify that the customers table contains 1200 rows
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers",
    database_connection
)["count"][0] == 1200

# Check 2: Verify that the products table contains 40 rows
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM products",
    database_connection
)["count"][0] == 40

# Check 3: Verify that the orders table contains 12309 rows
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders",
    database_connection
)["count"][0] == 12309

# Check 4: Verify that all order IDs are unique
assert pd.read_sql_query(
    "SELECT COUNT(DISTINCT order_id) AS count FROM orders",
    database_connection
)["count"][0] == 12309

# Check 5: Verify the number of completed orders
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='completed'",
    database_connection
)["count"][0] == 11110

# Check 6: Verify the number of cancelled orders
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='cancelled'",
    database_connection
)["count"][0] == 1199

# Check 7: Verify that no pending orders remain
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE status='pending'",
    database_connection
)["count"][0] == 0

# Check 8: Verify that no order amounts are missing
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM orders WHERE order_amount IS NULL",
    database_connection
)["count"][0] == 0

# Check 9: Verify that completed orders have delivery times
assert pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM orders
    WHERE status='completed' AND delivery_minutes IS NULL
    """,
    database_connection
)["count"][0] == 0

# Check 10: Verify that all cancelled orders have missing delivery times
assert pd.read_sql_query(
    """
    SELECT COUNT(*) AS count
    FROM orders
    WHERE status='cancelled' AND delivery_minutes IS NULL
    """,
    database_connection
)["count"][0] == 1199

# Check 11: Verify the total revenue from completed orders
completed_order_revenue = pd.read_sql_query(
    """
    SELECT SUM(order_amount) AS total
    FROM orders
    WHERE status='completed'
    """,
    database_connection
)["total"][0]

# Compare the calculated revenue with the expected revenue
assert round(completed_order_revenue, 2) == 2998668.80

# Check 12: Verify that no customer ages are missing
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers WHERE age IS NULL",
    database_connection
)["count"][0] == 0

# Check 13: Verify that no customer genders are missing
assert pd.read_sql_query(
    "SELECT COUNT(*) AS count FROM customers WHERE gender IS NULL",
    database_connection
)["count"][0] == 0

# Check 14: Verify the minimum customer age
assert pd.read_sql_query(
    "SELECT MIN(age) AS minimum FROM customers",
    database_connection
)["minimum"][0] == 18

# Check 15: Verify the maximum customer age
assert pd.read_sql_query(
    "SELECT MAX(age) AS maximum FROM customers",
    database_connection
)["maximum"][0] == 60

# STEP 7: CLOSE DATABASE CONNECTION
# Close the database connection after completing all validations
database_connection.close()

# FINAL SUCCESS MESSAGE
# Display this message only when all validation checkpoints pass
print("ALL CHECKPOINTS PASSED")