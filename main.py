
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()

selected_features = ["MDVP:Fo(Hz)", "MDVP:Jitter(%)"]
X = df[selected_features]
y = df["status"]

scaler = MinMaxScaler()
X_scaler = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaler, y, test_size=0.2, random_state=42
)



model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


import joblib

joblib.dump(model, 'my_model.joblib')
