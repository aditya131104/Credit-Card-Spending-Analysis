import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Credit Card Dashboard", layout="wide")

st.title("💳 Credit Card Spending Behavior Dashboard")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_transactions.csv")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['month'] = df['trans_date_trans_time'].dt.month

    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = 2026 - df['dob'].dt.year

    df['age_group'] = pd.cut(df['age'],
                            bins=[18,30,45,60,100],
                            labels=['18-30','30-45','45-60','60+'])
    return df

df = load_data()

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("Filters")

category_filter = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(df['category'].unique())
)

if category_filter != "All":
    df = df[df['category'] == category_filter]

# -------------------------------
# DATA PREVIEW
# -------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# -------------------------------
# CATEGORY SPENDING
# -------------------------------
st.subheader("🛒 Category-wise Spending")

category_spend = df.groupby('category')['amt'].sum().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
category_spend.plot(kind='bar', ax=ax1)
plt.xticks(rotation=90)

st.pyplot(fig1)

# -------------------------------
# MONTHLY TREND
# -------------------------------
st.subheader("📈 Monthly Spending Trend")

monthly_spend = df.groupby('month')['amt'].sum()

fig2, ax2 = plt.subplots()
monthly_spend.plot(kind='line', marker='o', ax=ax2)

st.pyplot(fig2)

# -------------------------------
# AGE GROUP ANALYSIS
# -------------------------------
st.subheader("👥 Spending by Age Group")

age_spend = df.groupby('age_group')['amt'].sum()

fig3, ax3 = plt.subplots()
age_spend.plot(kind='bar', ax=ax3)

st.pyplot(fig3)

# -------------------------------
# CUSTOMER SEGMENTATION
# -------------------------------
st.subheader("👑 Customer Segmentation")

customer_data = df.groupby('cc_num')['amt'].agg(['sum','mean','count'])
customer_data.columns = ['total_spend','avg_spend','frequency']

customer_data['segment'] = pd.qcut(customer_data['total_spend'],
                                   q=3,
                                   labels=['Low','Medium','High'])

st.dataframe(customer_data.head())

# -------------------------------
# HEATMAP
# -------------------------------
st.subheader("🔥 Heatmap (Month vs Category)")

pivot = df.pivot_table(values='amt', index='month', columns='category', aggfunc='sum')

fig4, ax4 = plt.subplots(figsize=(10,5))
sns.heatmap(pivot, ax=ax4)

st.pyplot(fig4)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("📌 Key Insights")

st.markdown("""
- Grocery and shopping categories generate the highest revenue  
- High-value customers contribute most of the spending  
- Spending peaks in mid-year months  
- Customers aged 30–45 are the most active spenders  
""")