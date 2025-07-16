#-----------------Task#01- Supervised Learning – Predicting House Prices (Project# 01)--------------------
# 1. Load the dataset and display the first five rows
from sklearn.datasets import fetch_california_housing
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print(df.head())
# 2. Explore the dataset
print("Samples:", df.shape[0])
print("Features:", df.shape[1] - 1)  # Exclude target column
print(df.describe())
print("hello")
# 3. Split the dataset:
from sklearn.model_selection import train_test_split
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Train a Linear Regression model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
# 5. Evaluate using MSE:
from sklearn.metrics import mean_squared_error

mse_lr = mean_squared_error(y_test, y_pred_lr)
print("Linear Regression MSE:", mse_lr)
# 6. Plot predicted vs actual prices
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred_lr, alpha=0.3)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()
# 7. Compare with Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Decision Tree MSE:", mse_tree)

#-----------------Task 2: Supervised Learning – Classifying Iris Flowers (Project# 02)-----------------

# 1. Load the dataset and display details
from sklearn.datasets import load_iris

iris = load_iris()
print("Features:", iris.feature_names)
print("Targets:", iris.target_names)

# 2. Visualize the dataset
import seaborn as sns

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['Species'] = iris.target_names[iris.target]
sns.scatterplot(data=df_iris, x='petal length (cm)', y='petal width (cm)', hue='Species')
plt.title('Iris Dataset: Petal Length vs Width')
plt.show()
# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
# 4. Train a KNN classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
# 5. Evaluate accuracy
from sklearn.metrics import accuracy_score

acc_knn = accuracy_score(y_test, y_pred_knn)
print("KNN Accuracy:", acc_knn)
# 6. Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred_knn)
sns.heatmap(cm, annot=True, xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for KNN')
plt.show()
# 7. Compare with Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
print("Logistic Regression Accuracy:", acc_logreg)
#---------------------Task 3: Unsupervised Learning – Customer Segmentation(Project# 03)------------------
# 1. Load the dataset
df_mall = pd.read_csv('Mall_Customers.csv')
print(df_mall.head())
# 2. Visualize Income vs Spending Score
plt.scatter(df_mall['Annual Income (k$)'], df_mall['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Data')
plt.show()
# 4. Use K-Means with optimal clusters (Elbow Method)
from sklearn.cluster import KMeans

X = df_mall[['Annual Income (k$)', 'Spending Score (1-100)']]
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()
# 5. Visualize clusters (using k=5)
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
df_mall['Cluster'] = clusters

sns.scatterplot(data=df_mall, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.show()
