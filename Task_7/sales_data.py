import sqlite3

conn =sqlite3.connect("sales_data.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sales(
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               product TEXT,
               quantity INTEGER,
               price REAL
               )
               """)

sales_data=[
    ("Phone ", 100, 10.99),
    ("Laptop", 50,  999.99),
    ("Tablet", 200,  299.99),
    ("Headphones", 150,  49.99),
    ("Charger", 250,50.99),
    ("Mouse", 300,  19.99),
    ("Keyboard", 400,  29.99),
    ("Monitor", 350,  199.99),
    ("Speaker", 450,  99.99),
    ("Camera", 550,  299.99),
    ("Printer", 650,  199.99),
    ("Scanner", 750,  99.99),
    ("Projector", 850,  499.99),
    ("Notebook", 950,  299.99),
    ("Desk", 1050,  199.99),
    ("Chair", 1150,  99.99),
    ("Table", 1250,  499.99),
    ("Shelf", 1350,  299.99),
    ("Bookcase", 1450,  199.99),
    ("Cabinet", 1550,  499.99),
    ("Filing Cabinet", 1650,  299.99),
    ("Ergonomic Chair", 1750,  199.99),
    ("Standing Desk", 1850,  499.99),
    ("Whiteboard", 1950,  299.99),
    ("Easel", 2050,  199.99),
    ("Pencil Case", 2150,  99.99),
    ("Pencil Sharpener", 2250,  49.99),
    ("Highlighter", 2350,  29.99),
    ("Calculator", 2450,  19.99),
    ("Ruler", 2550,  9.99),
    ("Pencil", 2650,  4.99),
    ("Eraser", 2750,  2.99),
    
]

cursor.executemany("INSERT INTO sales (product,quantity,price) VALUES (?,?,?)",sales_data)


conn.commit()

conn.close


import sqlite3
import pandas as pd
import matplotlib as plt

conn =sqlite3.Connect("sales_data.db")

qury ="""SELECT 
    product, 
    SUM(quantity) AS total_qty, 
    SUM(quantity * price) AS revenue 
FROM sales 
GROUP BY product;"""
df = pd.read_sql_query(qury, conn)

conn.close()

print("Sales Summary")
print(df)




df.plot(kind ='bar' , x = 'product' , y ='revenue', color = 'blue' , legend =False)


plt.title("Revenue by Product")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.tight_layout()

plt.show()