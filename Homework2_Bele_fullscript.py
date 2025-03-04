# import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn 
import matplotlib.pyplot as plt
file_path = "/Users/jcbele/Downloads/winequality.csv"
df = pd.read_csv(file_path)

print(df.head())

#Checking for missing values
print(df.isna().sum())

#Histogram for X values 
# Define the list of features to plot
features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
            'pH', 'sulphates', 'alcohol']

# Set up the figure and axes
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))  # Adjust rows/cols based on the number of features
axes = axes.flatten()  # Flatten to easily iterate

# Plot histograms for each feature
for i, feature in enumerate(features):
    sns.histplot(df[feature], bins=30, kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#Scatterplot of variables against quality
#features already defined
target = 'quality'

# Set up the figure and axes
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 12))  # Adjust rows/cols based on the number of features
axes = axes.flatten()  # Flatten to easily iterate

# Plot scatterplots for each feature against the target
for i, feature in enumerate(features):
    sns.scatterplot(x=df[feature], y=df[target], alpha=0.5, ax=axes[i])
    axes[i].set_title(f'{feature} vs {target}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel(target)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
#use cross-validation to find the best k
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
 
df['fixed acidity'].fillna(df['fixed acidity'].mean(), inplace=True)
df['citric acid'].fillna(df['citric acid'].mean(), inplace=True)
df['residual sugar'].fillna(df['residual sugar'].mean(), inplace=True)
df['free sulfur dioxide'].fillna(df['free sulfur dioxide'].mean(), inplace=True)
df['density'].fillna(df['density'].mean(), inplace=True)
df['pH'].fillna(df['pH'].mean(), inplace=True)
df['sulphates'].fillna(df['sulphates'].mean(), inplace=True)
df['quality'].fillna(df['quality'].mean(), inplace=True)

#now all missing values have been replaces using the method of means
#Based on the nature of the question, I feel it is necessary to include outliers to determine
## the best possible explanatory variable

# Transform all data using Min-Max Scaler to assimilate all data 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']] = scaler.fit_transform(df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']])

## It makes sense to me to assimilate this data in order to determine the best possible explanatory variable
df.to_csv('cleaned_winequality', index=False)

#Summary of new data set
## Instead of removing missing values, I replaced them with the mean depending on the column
## To keep data routine and easy to interpret, I used MinMax Scaling to assimilate the data


#first define X and y
X = df.drop(columns=['quality'])
y = df['quality']

# to determine a binary value for y, I will define a high quality wine as 7 or higher in the quality section
threshold = 6.1
y = (y >= threshold).astype(int)

#try different values of k
import matplotlib.pyplot as plt

# Try different values of k
k_values = range(1, 21)  # Values of k from 1 to 20
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5).mean()  # 5-fold cross-validation
    scores.append(score)

# Plot accuracy vs. k
plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o', linestyle='-', color='b', label="Cross-validation accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Cross-Validation Accuracy")
plt.title("k-NN Accuracy vs. k")
plt.xticks(k_values)
plt.legend()
plt.grid()
plt.show()

# Trainging and Evaluating kNN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the best k-NN model
best_k = 16
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Predict on test data
y_pred = knn.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#After cross-validation, I can conclude that the best classifier is k = 16 as shown in the graph
#After testing the data with the new found k, I can conclude that this is the most accurate way to predict high quality wine
#The confusion matrix 
