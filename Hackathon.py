import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv(r"C:\Users\Talal.TALALSLAPTOP\OneDrive\Desktop\participant_data.csv")

# Drop rows where 'choice' is NaN
df = df.dropna(subset=['choice'])

# Forward-fill missing values
df.ffill(inplace=True)

# Save a copy of original data to handle "no choice" later
original_df = df.copy()

# Encode categorical labels
le = LabelEncoder()
df['choice'] = le.fit_transform(df['choice'])
df['od'] = le.fit_transform(df['od'])

# Filter out "no choice" for training, keep only ADVS and PREF
df = df[df['choice'] != 2]

df['price_diff'] = df['PREF_price'] - df['ADVS_price']
df['price_ratio'] = df['PREF_price'] / (df['ADVS_price'] + 1)

# Fix occupancy calculation to avoid division by zero
df['ADVS_occupancy'] = (df['ADVS_capacity'] - df['ADVS_inventory']) / (df['ADVS_capacity'] + 1e-6)
df['PREF_occupancy'] = (df['PREF_capacity'] - df['PREF_inventory']) / (df['PREF_capacity'] + 1e-6)

features = ['trip_type', 'branded_fare', 'number_of_pax', 'ADVS_price', 'PREF_price',
            'ADVS_inventory', 'PREF_inventory', 'price_diff', 'price_ratio', 'ADVS_occupancy', 'PREF_occupancy']
X = df[features]
y = df['choice'] 

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True) # Replace NaN with zero


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert target to categorical
y_categorical = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Define Neural Network Model
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')  # Output layer
])

# Compile model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate performance
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
f1 = f1_score(y_test_labels, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

# Assign No Choice Customers to ADVS or PREF
no_choice_customers = original_df[original_df['choice'] == 2].copy()

# Apply same feature transformations
no_choice_customers['price_diff'] = no_choice_customers['PREF_price'] - no_choice_customers['ADVS_price']
no_choice_customers['price_ratio'] = no_choice_customers['PREF_price'] / (no_choice_customers['ADVS_price'] + 1)
no_choice_customers['ADVS_occupancy'] = (no_choice_customers['ADVS_capacity'] - no_choice_customers['ADVS_inventory']) / no_choice_customers['ADVS_capacity']
no_choice_customers['PREF_occupancy'] = (no_choice_customers['PREF_capacity'] - no_choice_customers['PREF_inventory']) / no_choice_customers['PREF_capacity']

# Scale features
if not no_choice_customers.empty:
    X_no_choice = scaler.transform(no_choice_customers[features])
    no_choice_probs = model.predict(X_no_choice)
    assigned_seats = np.argmax(no_choice_probs, axis=1)
else:
    print("No 'no_choice' customers found. Skipping assignment.")
    assigned_seats = np.array([]) 

# Assign seats in the original dataset
original_df.loc[original_df['choice'] == "no_choice", 'assigned_seat'] = assigned_seats

# Final assignment
original_df.loc[original_df['choice'] == "no_choice", 'assigned_seat'] = assigned_seats

# Convert assigned_seat back to labels
original_df['assigned_seat'] = original_df['assigned_seat'].map({0: 'ADVS', 1: 'PREF'})

print("\nAssigned 'No Choice' Customers to Seats")
print(original_df[original_df['choice'] == "no_choice"][['assigned_seat']].value_counts())
