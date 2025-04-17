select 
    product,
    SUM(quantity) as total_qty,
    SUM(quantity * price) as revenue
FROM sales
GROUP BY product;



