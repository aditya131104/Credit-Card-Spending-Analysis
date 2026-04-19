import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("credit_card_transactions.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)
print(df.head())


# -------------------------------
# 2. DATA CLEANING
# -------------------------------
df.drop_duplicates(inplace=True)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

print("\nMissing Values:\n", df.isnull().sum())


# -------------------------------
# 3. FEATURE ENGINEERING
# -------------------------------
df['year'] = df['trans_date_trans_time'].dt.year
df['month'] = df['trans_date_trans_time'].dt.month
df['day'] = df['trans_date_trans_time'].dt.day
df['hour'] = df['trans_date_trans_time'].dt.hour

df['dob'] = pd.to_datetime(df['dob'])
df['age'] = 2026 - df['dob'].dt.year

df['age_group'] = pd.cut(df['age'],
                        bins=[18,30,45,60,100],
                        labels=['18-30','30-45','45-60','60+'])


# -------------------------------
# 4. ANALYSIS
# -------------------------------

# Age group spending
age_spend = df.groupby('age_group')['amt'].sum()
print("\nSpending by Age Group:\n", age_spend)

# Category spending
category_spend = df.groupby('category')['amt'].sum().sort_values(ascending=False)
print("\nTop Categories:\n", category_spend.head())

# Monthly spending
monthly_spend = df.groupby('month')['amt'].sum()
print("\nMonthly Spending:\n", monthly_spend)


# -------------------------------
# 5. CUSTOMER SEGMENTATION
# -------------------------------
customer_data = df.groupby('cc_num')['amt'].agg(['sum','mean','count'])
customer_data.columns = ['total_spend','avg_spend','frequency']

customer_data['segment'] = pd.qcut(customer_data['total_spend'],
                                   q=3,
                                   labels=['Low','Medium','High'])

print("\nCustomer Segments:\n", customer_data.head())


# -------------------------------
# 6. VISUALIZATION
# -------------------------------

# Age group graph
age_spend.plot(kind='bar', title="Spending by Age Group")
plt.savefig("age_spending.png")
plt.close()

# Category graph
category_spend.plot(kind='bar', title="Category Spending")
plt.xticks(rotation=90)
plt.savefig("category_spending.png")
plt.close()

# Monthly trend
monthly_spend.plot(kind='line', marker='o', title="Monthly Spending")
plt.savefig("monthly_spending.png")
plt.close()

# Heatmap
pivot = df.pivot_table(values='amt', index='month', columns='category', aggfunc='sum')
sns.heatmap(pivot)
plt.title("Heatmap")
plt.savefig("heatmap.png")
plt.close()


# -------------------------------
# 7. SAVE OUTPUT
# -------------------------------
customer_data.to_csv("customer_segments.csv")
df.to_csv("processed_data.csv", index=False)


# -------------------------------
# 8. INSIGHTS
# -------------------------------
print("\nINSIGHTS:")
print("1. Some categories generate highest revenue.")
print("2. High-value customers contribute most spending.")
print("3. Spending peaks in certain months.")
print("4. Middle-age users spend more.")

import os
os.makedirs("output", exist_ok=True)

df.to_csv("output/processed_data.csv", index=False)
customer_data.to_csv("output/customer_segments.csv")

print("Files saved successfully in output folder ✅")