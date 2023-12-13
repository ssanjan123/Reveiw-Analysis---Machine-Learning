from keras.models import load_model
import pickle

# Load the model
loaded_model = load_model('my_sentiment_model.h5')
# Continue training
# loaded_model.fit(more_training_data, more_labels)

# # Or, make predictions
# predictions = loaded_model.predict(new_data)


with open('my_training_history.pkl', 'rb') as f:
    loaded_history = pickle.load(f)
