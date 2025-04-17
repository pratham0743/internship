import matplotlib as plt

df.plot(kind ='bar' , x = 'product' , y ='revenue', color = 'blue' , legend =False)


plt.title("Revenue by Product")
plt.xlabel("Product")
plt.ylabel("Revenue")
plt.tight_layout()

plt.show()