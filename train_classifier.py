import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print("Data shape:", data.shape)
print("Total labels:", len(labels))
print("Unique labels:", np.unique(labels))

x_train, x_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

model = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation='relu',
    solver='adam',
    max_iter=1000,
    verbose=True,
    random_state=42
)

print("\nTraining model...")
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {score * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))

with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels': np.unique(labels)}, f)

print("Model saved as model.p")
