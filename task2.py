import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

sns.set(style="whitegrid")

# Load CSV (same folder)
df = pd.read_csv(r"PRODIGY_DS_2\train (1).csv")

plt.figure(figsize=(18, 10))    

# 1. Age Distribution
plt.subplot(2, 3, 1)
sns.histplot(df['Age'].dropna(), bins=30, kde=True)
plt.title("Age Distribution")

# 2. Fare Distribution
plt.subplot(2, 3, 2)
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title("Fare Distribution")

# 3. Survival Count
plt.subplot(2, 3, 3)
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")

# 4. Passenger Class
plt.subplot(2, 3, 4)
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class")

# 5. Gender Distribution
plt.subplot(2, 3, 5)
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")

# 6. Embarked Port
plt.subplot(2, 3, 6)
sns.countplot(x='Embarked', data=df)
plt.title("Embarked Port")

plt.tight_layout()
plt.show()