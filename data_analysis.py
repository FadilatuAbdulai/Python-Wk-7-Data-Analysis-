# Task 1: Load and Explore the Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
try:
    iris = load_iris(as_frame=True)  # returns data as pandas DataFrame
    df = iris.frame  # combine features + target
    df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))  # add species name
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# Display first few rows
print("First 5 rows of dataset:")
print(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset (no missing values in Iris, but let's show method)
df = df.dropna()  # alternatively, df.fillna(method='ffill')

# Task 2: Basic Data Analysis
print("\nDescriptive Statistics:")
print(df.describe())

# Grouping: Mean of numerical columns per species
print("\nAverage measurements by species:")
print(df.groupby("species").mean())

# Task 3: Data Visualization
plt.style.use("seaborn-v0_8")

# Line chart (not time-series in Iris, so let's simulate with index)
plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
plt.title("Line Chart: Sepal Length over Index")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(7,5))
sns.barplot(x="species", y="petal length (cm)", data=df, estimator="mean")
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(7,5))
plt.hist(df["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(7,5))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df, palette="deep")
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()