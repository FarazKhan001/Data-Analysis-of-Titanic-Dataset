import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("titanic.csv")

# Basic Cleaning & Feature Engineering
df.drop_duplicates(inplace=True)
df.rename(columns={
    'Siblings/Spouses Aboard': 'Family_Aboard',
    'Parents/Children Aboard': 'Parents_Children',
    'Pclass': 'Passenger_Class',
    'Fare': 'Ticket_Fare',
    'Age': 'Passenger_Age',
    'Sex': 'Gender_Text',
    'Survived': 'Survival_Status'
}, inplace=True)
df['Family_Total'] = df['Family_Aboard'] + df['Parents_Children'] + 1

# Encode Age Groups and Fare Groups
df['AgeGroup'] = pd.cut(df['Passenger_Age'], [0, 15, 25, 45, 65, 80], labels=['Child', 'Youth', 'Adult', 'Senior', 'Elder'])
df['FareGroup'] = pd.cut(df['Ticket_Fare'], [0, 15, 40, 80, 600], labels=['Low', 'Medium', 'High', 'Very High'])

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender_Text'])

# Outlier Detection
# Fare Outliers
Q1_fare=df['Ticket_Fare'].quantile(0.25)
Q3_fare=df['Ticket_Fare'].quantile(0.75)
IQR_fare=Q3_fare-Q1_fare
fare_outliers=df[(df['Ticket_Fare'] < Q1_fare - 1.5 * IQR_fare) | (df['Ticket_Fare'] > Q3_fare + 1.5 * IQR_fare)]

# Age Outliers
Q1_age=df['Passenger_Age'].quantile(0.25)
Q3_age=df['Passenger_Age'].quantile(0.75)
IQR_age=Q3_age - Q1_age
age_outliers=df[(df['Passenger_Age'] < Q1_age - 1.5 * IQR_age) | (df['Passenger_Age'] > Q3_age + 1.5 * IQR_age)]

print("\n Number of Age Outliers:", age_outliers.shape[0])
print(age_outliers[['Name', 'Passenger_Age']].head())

print("\n Number of Fare Outliers:", fare_outliers.shape[0])
print(fare_outliers[['Name', 'Ticket_Fare']].head())

# Data Visualization
# Gender Proportion
plt.pie(df['Gender_Text'].value_counts(), labels=['Male', 'Female'], autopct='%1.1f%%', colors=['#66c2a5','#fc8d62'])
plt.title('Proportion of Genders')
plt.show()

# Pie Chart: Passenger Class Proportion
plt.pie(df['Passenger_Class'].value_counts(), labels=['Class 3','Class 1','Class 2'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Passenger Class Distribution')
plt.show()

# Bar Chart: Survival Count
sns.countplot(x='Survival_Status', data=df, palette='coolwarm')
plt.title('Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.show()

# Age vs Survival
sns.boxplot(x='Survival_Status', y='Passenger_Age', data=df, palette='Set3')
plt.title('Age Distribution by Survival')
plt.show()

# Average Fare by Class
plt.figure(figsize=(8, 6))
sns.barplot(x='Passenger_Class', y='Ticket_Fare', data=df, palette='magma', hue='Passenger_Class')
plt.title('Average Fare by Class')
plt.show()

# Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Passenger_Age', bins=30, kde=True, color='slateblue')
plt.title('Passenger Age Distribution')
plt.show()

# Fare Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='Ticket_Fare', bins=30, kde=True, color='darkorange')
plt.title('Ticket Fare Distribution')
plt.show()

# Gender Count by Class
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Passenger_Class', hue='Gender', palette='Set2')
plt.title('Passenger Count by Class and Gender')
plt.tight_layout()
plt.show()

# Violin Plot: Age vs Survival by Gender
plt.figure(figsize=(8, 6))
sns.violinplot(x='Survival_Status', y='Passenger_Age', hue='Gender_Text', data=df, split=True, palette='pastel')
plt.title('Age Distribution by Survival and Gender')
plt.show()

# Violin Plot: Fare vs Survival
plt.figure(figsize=(8, 6))
sns.violinplot(x='Survival_Status', y='Ticket_Fare', data=df, palette='Set3')
plt.title('Fare Distribution by Survival')
plt.show()

# Scatter Plot: Age vs Fare colored by Survival
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Passenger_Age', y='Ticket_Fare', hue='Survival_Status', palette='Set1')
plt.title('Age vs Fare by Survival')
plt.show()

# Scatter Plot: Age vs Fare with Gender
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Passenger_Age', y='Ticket_Fare', hue='Gender_Text', palette='cool')
plt.title('Age vs Fare by Gender')
plt.show()

# Pair Plot by Survival Status
numerical_cols = ['Passenger_Age', 'Ticket_Fare', 'Family_Aboard', 'Parents_Children']
sns.pairplot(df[numerical_cols + ['Survival_Status']], hue='Survival_Status', palette='Set2', diag_kind='kde')
plt.suptitle("Pair Plot of Titanic Features by Survival Status", y=1.02)
plt.show()

# Covariance Analysis
# Convert Gender to numeric for covariance calculation
df['Gender'] = df['Gender_Text'].map({'female': 0, 'male': 1})

# covariance 
cov_matrix = df[['Passenger_Class', 'Gender', 'Passenger_Age', 'Family_Aboard', 'Parents_Children', 'Ticket_Fare']].cov()
print("\n Covariance Matrix:\n", cov_matrix)

# Covariance Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cov_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Covariance Matrix Heatmap')
plt.show()

# Machine Learning Models

features = ['Passenger_Class', 'Gender', 'Passenger_Age', 'Family_Aboard', 'Parents_Children', 'Ticket_Fare']
X = df[features]
y = df['Survival_Status']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_preds)
print(f"\n KNN Accuracy: {knn_accuracy:.2f}")
print(f" KNN Predicted Survivors: {sum(knn_preds)} out of {len(knn_preds)}")
print(classification_report(y_test, knn_preds))

# Decision Tree 
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_preds)
print(f"\n Decision Tree Accuracy: {dt_accuracy:.2f}")
print(f" Decision Tree Predicted Survivors: {sum(dt_preds)} out of {len(dt_preds)}")
print(classification_report(y_test, dt_preds))

# Accuracy by K Plot
acc_scores = []
k_range = range(1, 21)
for k in k_range:
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    knn_loop.fit(X_train, y_train)
    preds = knn_loop.predict(X_test)
    acc_scores.append(accuracy_score(y_test, preds))

best_k = k_range[acc_scores.index(max(acc_scores))]

plt.plot(k_range, acc_scores, marker='o', color='red')
plt.title('KNN Accuracy by Number of Neighbors')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()