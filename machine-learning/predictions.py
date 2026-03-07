from sklearn.svm import LinearSVC;

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

label_mapping = {
  's': 'soluble',
  'n': 'not soluble'
}

for idx, prediction in enumerate(predictions):
    print(f"Test data {idx+1}: {label_mapping[prediction]}")


