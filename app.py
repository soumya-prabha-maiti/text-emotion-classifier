import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import numpy as np

# Class Labels
LABELS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
NO_OF_CLASSES = len(LABELS)

# Path to the model weights
MODEL_PATH = "emotion_classifier.h5"

# Load the tokenizer
tokenizer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")

# Load the BERT model
bert_model = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

# Load the model with custom objects
classifier = tf.keras.models.load_model(MODEL_PATH,custom_objects={"KerasLayer": bert_model})

# Define the emotion classification function
def classify_sentiment(input_text):
    # Predict the emotion probabilities of the input text
    output_probabilities = np.squeeze(classifier.predict([input_text]))

    # Return the emotion score as a dictionary
    scores = {label: float(output_probabilities[idx]) for idx, label in enumerate(LABELS)}

    # return scores
    return scores


# Define the Gradio interface
iface = gr.Interface(
    fn=classify_sentiment,
    inputs=gr.Textbox(lines=5, label="Input Text"),
    outputs=gr.Label(),
    title="Text Emotion Classification",
    description="Enter a text and get the probability of it belonging to each emotion category.",
    examples=[
        ["Wow, this is amazing!"],
        ["This movie is a waste of time."],
        ["I am sad and have no energy to do any work."],
    ],
)

# Launch the interface
iface.launch()