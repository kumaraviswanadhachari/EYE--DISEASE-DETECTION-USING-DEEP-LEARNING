import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  

# Define categories (replace with your actual labels)
CATEGORIES = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# Function to extract basic image features
def extract_features(image_path):
    return features

# Load images and labels
data = []
labels = []

dataset_path = "dataset/"  

for category in CATEGORIES:
    folder_path = os.path.join(dataset_path, category)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        data.append(extract_features(img_path))
        labels.append(CATEGORIES.index(category))

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "eye_disease_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
