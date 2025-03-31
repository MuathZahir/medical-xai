import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from imblearn.over_sampling import SMOTE

# 1. Load the saved model
model = load_model('tcn_final_model.h5')  # Replace with your model's filename

# 2. Load your data (assuming it's in 'deduplicated_data_.csv')
df = pd.read_csv('Dataset/deduplicated_data_.csv')

# 3. Preprocessing: Create sequences
def create_sequences(data, user_code, window_size=10):
    sequences = []
    labels = []

    user_data = data[data['user_code'] == user_code].sort_values('date')

    if len(user_data) <= window_size:
        return [], []

    for i in range(len(user_data) - window_size):
        X = user_data.iloc[i:i+window_size].drop(['date', 'user_code', 'target', 'days_to_symptoms'], axis=1).values
        y = user_data.iloc[i+window_size]['target']

        sequences.append(X)
        labels.append(y)

    return sequences, labels

# 4. Process all users to create sequences and labels
all_sequences = []
all_labels = []
for user in df['user_code'].unique():
    seq, lab = create_sequences(df, user)
    all_sequences.extend(seq)
    all_labels.extend(lab)

X = np.array(all_sequences)
y = np.array(all_labels)

# 5. Normalization (using previously calculated means and stds)
feature_means = np.mean(X, axis=(0, 1))  # You might need to load these from a file
feature_stds = np.std(X, axis=(0, 1))    # if they were saved separately
feature_stds[feature_stds == 0] = 1       # Avoid division by zero
X_normalized = (X - feature_means) / feature_stds

# # 6. Reshape for SMOTE
# X_shape = X_normalized.shape
# X_reshaped = X_normalized.reshape(X_normalized.shape[0], -1)

# # 7. Apply SMOTE
# smote = SMOTE(random_state=42)  # Use the same random_state as during training
# X_smote, y_smote = smote.fit_resample(X_reshaped, y)

# # 8. Reshape back for the model
# X_smote = X_smote.reshape(X_smote.shape[0], X_shape[1], X_shape[2])

# 9. Now you can use the model to make predictions
predictions = model.predict(X_normalized)

# ... (further processing of predictions, e.g., applying threshold) ...