-- List all customers from USA, sorted by name
SELECT customerName, country 
FROM customers 
WHERE country = 'USA' 
ORDER BY customerName;


SELECT customerName, country 
FROM customers 
WHERE country = 'CANADA' 
ORDER BY customerName;


-- Total payment per customer
SELECT customerNumber, SUM(amount) AS total_spent 
FROM payments 
GROUP BY customerNumber 
ORDER BY total_spent DESC;

-- Orders with customer names
SELECT o.orderNumber, c.customerName, o.status 
FROM orders o
JOIN customers c ON o.customerNumber = c.customerNumber;

-- Customers and their orders (even those without orders)
SELECT c.customerName, o.orderNumber 
FROM customers c
LEFT JOIN orders o ON c.customerNumber = o.customerNumber;

-- Customers who placed orders above average total amount
SELECT customerName 
FROM customers 
WHERE customerNumber IN (
    SELECT customerNumber 
    FROM payments 
    GROUP BY customerNumber 
    HAVING SUM(amount) > (
        SELECT AVG(amount) FROM payments
    )
);


-- View of total payments per customer
CREATE VIEW customer_payments1 AS
SELECT customerNumber, SUM(amount) AS total_paid
FROM payments
GROUP BY customerNumber;

-- Index for faster searching on orderNumber
CREATE INDEX idx_orderNumber ON orders(orderNumber);

