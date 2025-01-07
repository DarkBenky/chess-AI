import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling1D
from tensorflow.keras.layers import Add, MultiHeadAttention
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
df = pd.read_csv('games.csv')

# Prepare input and output
X = df['X'].apply(eval).tolist()
Y = df['Y'].apply(lambda x: x if isinstance(x, list) else [0, 0, 0]).tolist()

# Convert to numpy arrays instead of TensorFlow tensors
X = np.array(X, dtype=np.int32)  # Changed to int32 for Embedding layer
Y = np.array(Y, dtype=np.int32)

# One-hot encode the labels (assuming promotion has 4 possible values: [0,1,2,3])
Y_from = to_categorical(Y[:, 0], num_classes=64)
Y_to = to_categorical(Y[:, 1], num_classes=64)
Y_promo = to_categorical(Y[:, 2], num_classes=5)  # Including no promotion

# Concatenate the outputs using NumPy
Y_final = np.concatenate([Y_from, Y_to, Y_promo], axis=1)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_final, test_size=0.2, random_state=42
)

def create_model():
    dim = 128  # Base dimension for network
    inputs = Input(shape=(64,), dtype='int32')
    
    # Embedding matching transformer dimension
    x = Embedding(input_dim=17, output_dim=dim)(inputs)
    
    # 4 transformer blocks (~50M params each)
    for _ in range(4):
        # Multi-head attention with matching dimensions
        attention = MultiHeadAttention(
            num_heads=16,
            key_dim=dim
        )(x, x)
        x = Add()([x, attention])
        
        # Dense feedforward with matching dimensions
        ffn = Dense(dim*2, activation='relu')(x)
        ffn = Dense(dim, activation='relu')(ffn)
        x = Add()([x, ffn])
        x = Dropout(0.1)(x)
    
    # Final processing
    x = GlobalAveragePooling1D()(x)
    x = Dense(dim*2, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(133, activation='softmax')(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = create_model()
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    X_train, Y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test Accuracy: {accuracy:.2f}')