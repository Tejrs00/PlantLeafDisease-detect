import tensorflow as tf

try:
    model = tf.keras.models.load_model("trained_model.keras")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading .keras model:", e)

try:
    model = tf.keras.models.load_model("trained_model.h5")
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading .h5 model:", e)
