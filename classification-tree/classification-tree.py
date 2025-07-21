import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv('titanic.csv')
df = df.drop(columns=['Name'])

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# Define features and target
X = df.drop(columns=['Survived'])
Y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train an XGBoost Classifier
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = xgb_model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print(f"XGBoost Classifier Accuracy: {accuracy:.2%}")

# Display the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.show()