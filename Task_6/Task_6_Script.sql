CREATE TABLE orders_detailed (
    order_id VARCHAR(10),
    order_date DATE,
    customer_id VARCHAR(10),
    product_id VARCHAR(10),
    category VARCHAR(50),
    quantity INT,
    unit_price DECIMAL(10,2),
    discount DECIMAL(4,2),
    region VARCHAR(20)
);
-- Monthly Revenue + Order Volume


SELECT
  YEAR(order_date) AS year,
  MONTH(order_date) AS month,
  SUM(quantity * unit_price * (1 - discount)) AS monthly_revenue,
  COUNT(DISTINCT order_id) AS order_volume
FROM
  orders_detailed
GROUP BY
  YEAR(order_date),
  MONTH(order_date)
ORDER BY
  year, month;

-- Monthly Trend by Region


SELECT
  YEAR(order_date) AS year,
  MONTH(order_date) AS month,
  region,
  SUM(quantity * unit_price * (1 - discount)) AS monthly_revenue,
  COUNT(DISTINCT order_id) AS order_volume
FROM
  orders_detailed
GROUP BY
  YEAR(order_date), MONTH(order_date), region
ORDER BY
  year, month, region;
  
  
-- Top 3 Months by Revenue
  
  
  SELECT
  YEAR(order_date) AS year,
  MONTH(order_date) AS month,
  SUM(quantity * unit_price * (1 - discount)) AS monthly_revenue
FROM
  orders_detailed
GROUP BY
  YEAR(order_date), MONTH(order_date)
ORDER BY
  monthly_revenue DESC
LIMIT 3;


