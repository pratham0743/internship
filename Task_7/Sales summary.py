import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

conn =sqlite3.connect("sales_data.db")

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