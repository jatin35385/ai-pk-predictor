import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Generate data
def generate_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'gender': np.random.randint(0, 2, n),
        'weight': np.random.normal(70, 10, n),
        'liver_score': np.random.normal(30, 5, n),
        'creatinine': np.random.normal(1.0, 0.2, n)
    })

    def calc_half_life(row):
        return 0.7 * row['weight'] / (row['creatinine'] + 0.5)

    def calc_clearance(row):
        return max(0.1, row['liver_score'] * 0.12 - row['creatinine'] * 1.5)

    def calc_auc(row):
        return 500 / (calc_clearance(row) + 5)

    data['half_life'] = data.apply(calc_half_life, axis=1)
    data['clearance'] = data.apply(calc_clearance, axis=1)
    data['auc'] = data.apply(calc_auc, axis=1)

    return data

data = generate_data(1000)

# 2. Prepare inputs and outputs
features = ['age', 'gender', 'weight', 'liver_score', 'creatinine']
targets = ['half_life', 'clearance', 'auc']

X = data[features]
y = data[targets]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 4. Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# 5. Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.4f}")

# 6. Plot training loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 7. Make prediction for new patient
new_patient = np.array([[25, 1, 68, 28, 1.2]])  # Change these values as needed
new_scaled = scaler.transform(new_patient)
prediction = model.predict(new_scaled)

print("\n📊 Prediction for New Patient:")
print(f"Half-life:   {prediction[0][0]:.2f} hours")
print(f"Clearance:   {prediction[0][1]:.2f} L/hour")
print(f"AUC:         {prediction[0][2]:.2f} µg·hr/mL")
