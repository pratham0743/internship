# %%
import pandas as pd 
import numpy as np
import seaborn as sns


# %%
df = pd.read_csv('D:/DA/Project/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()


# %%
df.info
df.isnull().sum
df.duplicated().sum

# %%
df.describe

df['Attrition'].value_counts()
df['Attrition'].value_counts(normalize=True)

# %%
print("Dataset Loaded Successfully")
print("Shape",df.shape)
print(df.head())


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_csv('D:/DA/Project/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()

# %%
plt.figure(figsize=(8,4))
sns.countplot(data=df,x='Department',hue='Attrition')
plt.title("Attrition Count by Department")
plt.xticks(rotation=45)
plt.show

# %%
sns.boxplot(x='Attrition', y='MonthlyIncome',data=df)
plt.title("Salary vs Attrition")


# %%
sns.countplot(x='YearsSinceLastPromotion',hue='Attrition',data=df)
plt.title("Promotion vs Attrition")

# %%
num_cols = df.select_dtypes(include='number').columns
plt.figure(figsize=(10,6))
sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",cmap='coolwarm')
plt.title('correlation Heatmap')
plt.tight_layout()
plt.show()


# %%
from sklearn.preprocessing import OneHotEncoder,StandardScaler


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# %%
df = pd.read_csv('D:/DA/Project/WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head()

# %%
from matplotlib import axis


x=df.drop("Attrition",axis=1)
y=df["Attrition"].map({'Yes':1,'No':0})


# %%
import select

from django.urls import include


cat_cols=x.select_dtypes(include='object').columns
num_cols=x.select_dtypes(include='number').columns


# %%
ohe= OneHotEncoder(sparse_output=False,drop='first')
x_cat=pd.DataFrame(ohe.fit_transform(x[cat_cols]),index=x.index)


# %%
scaler=StandardScaler()
x_num=pd.DataFrame(scaler.fit_transform(x[num_cols]),index=x.index,columns=num_cols)


# %%
x_processed=pd.concat([x_num,x_cat],axis=1)
print(f"Processed feature matrix:{x_processed.shape}")

# %%
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(
    x_processed,y,test_size=0.2,random_state=42,stratify=y)
print(f"train_test_split:{x_train.shape[0]}/{x_test.shape[0]}rows")

# %%
print("Train attrition ratio:", y_train.mean())
print("Test  attrition ratio:", y_test.mean())


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

x_train.columns = x_train.columns.astype(str)
x_test.columns = x_test.columns.astype(str)

results={}
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    results[name] = y_pred
    print(f"Trained {name}")


# %%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

metrics = {}
for name, y_pred in results.items():
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    metrics[name] = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, cm=cm)
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.2%}, Precision: {prec:.2%}, Recall: {rec:.2%}, F1: {f1:.2%}")
    print("Confusion Matrix:\n", cm)



import shap


best_name = max(metrics, key=lambda n: metrics[n]['f1'])
best_model = models[best_name]


explainer = shap.Explainer(best_model, x_train)   
shap_values = explainer(x_test)                   


shap.summary_plot(shap_values, x_test)



dept_attrition = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack().fillna(0)

df.to_csv('D:/DA/Project/HR_full_data.csv', index=False)
dept_attrition.reset_index().to_csv('D:/DA/Project/dept_attrition.csv', index=False)
import numpy as np

def convert_ndarray_to_list(d):
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = convert_ndarray_to_list(value)  
    elif isinstance(d, np.ndarray):
        return d.tolist()  
    return d  


metrics = convert_ndarray_to_list(metrics)


import json
with open('D:/DA/Project/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Data exported successfully for Power BI and reporting.")

import shap
import numpy as np
import pandas as pd

# Step 1: Create explainer
explainer = shap.TreeExplainer(model)

# Step 2: Get SHAP values
shap_values = explainer.shap_values(x_train)

# Step 3: Use shap_values[1] → positive class ("Yes" for attrition)
shap_df = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': np.abs(shap_values[1]).mean(axis=0)  # Mean absolute SHAP values
})

# Step 4: Sort top 10 and export
shap_df = shap_df.sort_values(by='Importance', ascending=False).head(10)
shap_df.to_csv('top_10_shap_features.csv', index=False)
