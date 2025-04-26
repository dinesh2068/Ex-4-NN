import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("Iris_data.csv")
df.columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df = df.dropna()

a = df.iloc[:, 0:4]
b = df.iloc[:, 4]

training_a, testing_a, training_b, testing_b = train_test_split(a, b, test_size=0.25, random_state=42)

myscaler = StandardScaler()
training_a = myscaler.fit_transform(training_a)
testing_a = myscaler.transform(testing_a)

m1 = MLPClassifier(hidden_layer_sizes=(12, 13, 14), activation='relu', solver='adam', max_iter=2500)
m1.fit(training_a, training_b)

predicted_values = m1.predict(testing_a)

print(confusion_matrix(testing_b, predicted_values))
print(classification_report(testing_b, predicted_values))
