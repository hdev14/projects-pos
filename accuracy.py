from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


data = [
  [1,1,1],
  [0,0,0],
  [1,0,1],
  [0,1,0],
  [1,1,0],
  [0,0,1],
]

labels = ['s', 'n', 's', 'n', 's', 's']

model = LinearSVC()
model.fit(data, labels)


test_data = [
  [1,0,0],
  [0,1,1],
  [1,0,1]
]

predictions = model.predict(test_data)

expected_labels = ['s', 's', 's']

accuracy = accuracy_score(expected_labels, predictions) * 100

print(f"Accuracy: {accuracy:.2f}%")


