-- ============================================================
-- TASK 6: SQL ANALYSIS
-- Instamart Customer Intelligence & Support Copilot
-- ============================================================


-- Q1: WHERE + GROUP BY + HAVING
-- Customers with completed orders above INR 500
-- and qualifying total spend above INR 8000

SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- ============================================================
-- Q2: THREE-TABLE JOIN
-- Revenue by city and product category

SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, revenue DESC;


-- ============================================================
-- Q3: SCALAR SUBQUERY
-- Products priced above average catalog price

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- ============================================================
-- Q4: LEFT JOIN
-- Customers without completed orders

SELECT
    c.customer_id,
    c.city,
    c.gender
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- ============================================================
-- Q5: CTE
-- Customers spending above average customer spend


WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT
        AVG(total_spend) AS average_spend
    FROM customer_spend
)
SELECT
    customer_id,
    total_spend,
    average_spend
FROM customer_spend
CROSS JOIN avg_spend
WHERE total_spend > average_spend
ORDER BY total_spend DESC;



-- ============================================================
-- Q6: WINDOW FUNCTIONS
-- Rank customers by completed spending within each city

WITH customer_spend AS (
    SELECT
        c.city,
        o.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM orders o
    JOIN customers c
        ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, o.customer_id
)
SELECT
    city,
    customer_id,
    total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_num
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- ============================================================
-- Q7: WINDOW FUNCTION
-- Daily revenue and cumulative revenue

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_date,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (
    ORDER BY order_date
) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_date;