# ==========================================
# TOURISM DATA ANALYSIS AND RECOMMENDATION
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
 mean_absolute_error,
 mean_squared_error,
 r2_score)
# ==========================================
# LOAD DATASET
# ==========================================
df = pd.read_csv(r"C:\Users\Preksha MV\Downloads\tourism_dataset.csv")
print("Dataset Shape:", df.shape)
print("\nFirst 5 Records")
print(df.head())
print("\nMissing Values")
print(df.isnull().sum())
# ==========================================
# DATA CLEANING
# ==========================================
df.drop_duplicates(inplace=True)
# ==========================================
# VISUALIZATION 1
# AVERAGE REVENUE BY CATEGORY
# ==========================================
avg_revenue = df.groupby("Category")["Revenue"].mean()
plt.figure(figsize=(8,5))
avg_revenue.plot(kind="bar")
plt.title("Average Revenue by Tourism Category")
plt.xlabel("Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.show()
# ==========================================
# VISUALIZATION 2
# CATEGORY DISTRIBUTION PIE CHART
# ==========================================
category_count = df["Category"].value_counts()
plt.figure(figsize=(7,7))
plt.pie(
 category_count,
 labels=category_count.index,
 autopct="%1.1f%%"
)
plt.title("Tourism Category Distribution")
plt.show()
# ==========================================
# VISUALIZATION 3
# AVERAGE REVENUE BY COUNTRY
# ==========================================
country_revenue = (
 df.groupby("Country")["Revenue"]
 .mean()
 .sort_values(ascending=False)
)
plt.figure(figsize=(10,5))
plt.plot(
 country_revenue.index,
 country_revenue.values,
 marker='o'
)
plt.title("Average Revenue by Country")
plt.xlabel("Country")
plt.ylabel("Revenue")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# ==========================================
# MACHINE LEARNING
# REVENUE PREDICTION
# ==========================================
ml_df = df.copy()
ml_df = pd.get_dummies(
 ml_df,
 columns=[
 "Country",
 "Category",
 "Accommodation_Available"
 ],
 drop_first=True
)
X = ml_df.drop(
 ["Revenue", "Location"],
 axis=1
)
y = ml_df["Revenue"]
X_train, X_test, y_train, y_test = train_test_split(
 X,
 y,
 test_size=0.2,
 random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# ==========================================
# MODEL EVALUATION
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(
 mean_squared_error(y_test, y_pred)
)
r2 = r2_score(y_test, y_pred)
print("\n===== MODEL PERFORMANCE =====")
print("MAE :", round(mae,2))
print("RMSE:", round(rmse,2))
print("R² Score:", round(r2,4))
# ==========================================
# RECOMMENDATION SYSTEM
# ==========================================
print("\n===== TOURIST RECOMMENDATION =====")
category = input(
 "Enter Category (Beach, Nature, Adventure, Cultural, Historical, Urban): "
)
rating = float(
 input("Enter Minimum Rating (1-5): ")
)
recommendations = df[
 (df["Category"].str.lower() == category.lower())
 &
 (df["Rating"] >= rating)
]
recommendations = recommendations.sort_values(
 by="Rating",
 ascending=False
)
print("\nRecommended Destinations:")
if len(recommendations) == 0:
 print("No matching destinations found.")
else:
 print(
 recommendations[
 [
 "Location",
 "Country",
 "Category",
 "Rating",
 "Revenue"
 ]
 ].head(10) )