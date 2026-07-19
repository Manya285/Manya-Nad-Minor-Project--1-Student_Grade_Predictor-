import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("../Data/student_data.csv")

print("Dataset Loaded Successfully!")
print(data.head())


# Convert grades into numbers
grade_mapping = {
    "A": 3,
    "B": 2,
    "C": 1,
    "D": 0
}

data["final_grade"] = data["final_grade"].map(grade_mapping)


# Features and target
X = data.drop("final_grade", axis=1)
y = data["final_grade"]


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create model
model = DecisionTreeClassifier()


# Train model
model.fit(X_train, y_train)


# Test model
prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("Model Accuracy:", accuracy)


# Predict a new student's grade

new_student = [[6, 85, 80, 85]]

result = model.predict(new_student)


reverse_mapping = {
    3: "A",
    2: "B",
    1: "C",
    0: "D"
}

print("Predicted Grade:", reverse_mapping[result[0]])