-- ============================================================
-- Enterprise AI Data Analyst — Analytical SQL Queries (12)
-- Member 1: Data Engineering & SQL Lead
-- Uses: JOIN, CTE, Window Functions, Aggregation
-- ============================================================

-- 1. Top 10 customers by total spend (JOIN + GROUP BY)
SELECT
    o.customer_id,
    c.country,
    ROUND(SUM(oi.revenue), 2) AS total_spent
FROM order_items oi
JOIN orders o    ON oi.invoice_no = o.invoice_no
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.is_cancelled = 0
GROUP BY o.customer_id, c.country
ORDER BY total_spent DESC
LIMIT 10;


-- 2. Monthly revenue trend
SELECT
    strftime('%Y-%m', o.invoice_date) AS year_month,
    ROUND(SUM(oi.revenue), 2)         AS monthly_revenue
FROM order_items oi
JOIN orders o ON oi.invoice_no = o.invoice_no
WHERE o.is_cancelled = 0
GROUP BY year_month
ORDER BY year_month;


-- 3. Running (cumulative) revenue total over time — WINDOW FUNCTION
WITH monthly AS (
    SELECT
        strftime('%Y-%m', o.invoice_date) AS year_month,
        SUM(oi.revenue)                   AS monthly_revenue
    FROM order_items oi
    JOIN orders o ON oi.invoice_no = o.invoice_no
    WHERE o.is_cancelled = 0
    GROUP BY year_month
)
SELECT
    year_month,
    ROUND(monthly_revenue, 2) AS monthly_revenue,
    ROUND(SUM(monthly_revenue) OVER (ORDER BY year_month), 2) AS running_total
FROM monthly
ORDER BY year_month;


-- 4. Rank customers by spend within each country — WINDOW FUNCTION (PARTITION BY)
WITH customer_spend AS (
    SELECT
        c.country,
        o.customer_id,
        SUM(oi.revenue) AS total_spent
    FROM order_items oi
    JOIN orders o    ON oi.invoice_no = o.invoice_no
    JOIN customers c ON o.customer_id = c.customer_id
    WHERE o.is_cancelled = 0
    GROUP BY c.country, o.customer_id
)
SELECT
    country,
    customer_id,
    ROUND(total_spent, 2) AS total_spent,
    RANK() OVER (PARTITION BY country ORDER BY total_spent DESC) AS rank_in_country
FROM customer_spend
ORDER BY country, rank_in_country;


-- 5. First purchase date per customer (CTE) + customer lifetime in days
WITH first_purchase AS (
    SELECT
        customer_id,
        MIN(invoice_date) AS first_order_date,
        MAX(invoice_date) AS last_order_date
    FROM orders
    WHERE is_cancelled = 0
    GROUP BY customer_id
)
SELECT
    customer_id,
    first_order_date,
    last_order_date,
    CAST(julianday(last_order_date) - julianday(first_order_date) AS INTEGER) AS lifetime_days
FROM first_purchase
ORDER BY lifetime_days DESC
LIMIT 20;


-- 6. Top 10 best-selling products by quantity
SELECT
    p.stock_code,
    p.description,
    SUM(oi.quantity) AS total_units_sold
FROM order_items oi
JOIN products p ON oi.stock_code = p.stock_code
JOIN orders o   ON oi.invoice_no = o.invoice_no
WHERE o.is_cancelled = 0
GROUP BY p.stock_code, p.description
ORDER BY total_units_sold DESC
LIMIT 10;


-- 7. Cancellation rate per month
SELECT
    strftime('%Y-%m', invoice_date) AS year_month,
    COUNT(*)                                          AS total_orders,
    SUM(is_cancelled)                                  AS cancelled_orders,
    ROUND(100.0 * SUM(is_cancelled) / COUNT(*), 2)     AS cancellation_rate_pct
FROM orders
GROUP BY year_month
ORDER BY year_month;


-- 8. Average order value (AOV) per country
WITH order_totals AS (
    SELECT
        o.invoice_no,
        o.customer_id,
        SUM(oi.revenue) AS order_value
    FROM order_items oi
    JOIN orders o ON oi.invoice_no = o.invoice_no
    WHERE o.is_cancelled = 0
    GROUP BY o.invoice_no, o.customer_id
)
SELECT
    c.country,
    ROUND(AVG(ot.order_value), 2) AS avg_order_value,
    COUNT(ot.invoice_no)          AS num_orders
FROM order_totals ot
JOIN customers c ON ot.customer_id = c.customer_id
GROUP BY c.country
ORDER BY avg_order_value DESC;


-- 9. One-time buyers (bought exactly once) — simple churn/loyalty indicator
WITH order_counts AS (
    SELECT customer_id, COUNT(DISTINCT invoice_no) AS num_orders
    FROM orders
    WHERE is_cancelled = 0
    GROUP BY customer_id
)
SELECT
    COUNT(*) AS one_time_buyers,
    (SELECT COUNT(*) FROM order_counts) AS total_customers,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM order_counts), 2) AS pct_one_time_buyers
FROM order_counts
WHERE num_orders = 1;


-- 10. Month-over-month revenue growth % — WINDOW FUNCTION (LAG)
WITH monthly AS (
    SELECT
        strftime('%Y-%m', o.invoice_date) AS year_month,
        SUM(oi.revenue)                   AS monthly_revenue
    FROM order_items oi
    JOIN orders o ON oi.invoice_no = o.invoice_no
    WHERE o.is_cancelled = 0
    GROUP BY year_month
)
SELECT
    year_month,
    ROUND(monthly_revenue, 2) AS monthly_revenue,
    ROUND(monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year_month), 2) AS revenue_change,
    ROUND(
        100.0 * (monthly_revenue - LAG(monthly_revenue) OVER (ORDER BY year_month))
        / NULLIF(LAG(monthly_revenue) OVER (ORDER BY year_month), 0), 2
    ) AS mom_growth_pct
FROM monthly
ORDER BY year_month;


-- 11. Top 5 products per country by revenue — WINDOW FUNCTION (ROW_NUMBER + PARTITION)
WITH product_country_sales AS (
    SELECT
        c.country,
        p.stock_code,
        p.description,
        SUM(oi.revenue) AS revenue
    FROM order_items oi
    JOIN orders o    ON oi.invoice_no = o.invoice_no
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p  ON oi.stock_code = p.stock_code
    WHERE o.is_cancelled = 0
    GROUP BY c.country, p.stock_code, p.description
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY country ORDER BY revenue DESC) AS rn
    FROM product_country_sales
)
SELECT country, stock_code, description, ROUND(revenue, 2) AS revenue
FROM ranked
WHERE rn <= 5
ORDER BY country, rn;


-- 12. Customer segments by spend quartile — WINDOW FUNCTION (NTILE)
WITH customer_spend AS (
    SELECT
        o.customer_id,
        SUM(oi.revenue) AS total_spent
    FROM order_items oi
    JOIN orders o ON oi.invoice_no = o.invoice_no
    WHERE o.is_cancelled = 0
    GROUP BY o.customer_id
)
SELECT
    customer_id,
    ROUND(total_spent, 2) AS total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS spend_quartile
FROM customer_spend
ORDER BY total_spent DESC;
