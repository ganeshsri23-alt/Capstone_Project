import sqlite3
import pandas as pd

# Connect to the Instamart database
database_connection = sqlite3.connect("data/instamart.db")

# Read all the SQL queries from the file
with open("sql/queries.sql", "r", encoding="utf-8") as sql_file:
    sql_script = sql_file.read()

# Remove SQL comment lines so semicolons inside comments
# do not accidentally split the queries
sql_script = "\n".join(
    line for line in sql_script.splitlines()
    if not line.strip().startswith("--")
)

# Split the SQL script into individual queries
sql_queries = sql_script.split(";")

# Run each query one by one and display its result
for query_number, sql_query in enumerate(sql_queries, start=1):
    if sql_query.strip():
        print(f"\n========== QUERY {query_number} ==========")
        query_result = pd.read_sql_query(
            sql_query,
            database_connection
        )
        print(query_result)

# Close the database connection after running all queries
database_connection.close()