---
title: Text Emotion Classifier using BERT
emoji: 📊
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 3.35.2
app_file: app.py
pinned: false
license: mit
---

# Text Emotion Classifier using BERT

This project is a text emotion classifier that can be used to classify the emotion of a given text into one of the following categories: anger, fear, joy, love, sadness, surprise. The model is built using the [BERT](https://arxiv.org/abs/1810.04805) architecture and is trained on the [Emotion Dataset](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp) from Kaggle.

## Demo

The deployed version of this project can be accessed at [Hugging Face Spaces](https://huggingface.co/spaces/soumyaprabhamaiti/text_emotion_classifier_using_BERT).

## Installing Locally

To run this project locally, please follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/soumya-prabha-maiti/text-emotion-classifier
   ```

2. Navigate to the project folder:

   ```
   cd text-emotion-classifier
   ```

3. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

4. Run the application:

   ```
   python app.py
   ```

5. Access the application in your web browser at the specified port.

## Dataset

The dataset used in this project is the [Emotion Dataset](https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp) from Kaggle. The dataset consists of sentences labeled with one of the following emotions: anger, fear, joy, love, sadness, surprise. The dataset is split into training, validation and testing sets with 16000, 2000 and 2000 sentences respectively.

## Model Architecture

The sentiment classifier model has the following architecture:

```
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 text_input (InputLayer)        [(None,)]            0           []                               
                                                                                                  
 keras_layer (KerasLayer)       {'input_word_ids':   0           ['text_input[0][0]']             
                                (None, 128),                                                      
                                 'input_type_ids':                                                
                                (None, 128),                                                      
                                 'input_mask': (Non                                               
                                e, 128)}                                                          
                                                                                                  
 keras_layer_1 (KerasLayer)     {'sequence_output':  109482241   ['keras_layer[0][0]',            
                                 (None, 128, 768),                'keras_layer[0][1]',            
                                 'encoder_outputs':               'keras_layer[0][2]']            
                                 [(None, 128, 768),                                               
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768),                                                
                                 (None, 128, 768)],                                               
                                 'pooled_output': (                                               
                                None, 768),                                                       
                                 'default': (None,                                                
                                768)}                                                             
                                                                                                  
 dropout (Dropout)              (None, 768)          0           ['keras_layer_1[0][13]']         
                                                                                                  
 output (Dense)                 (None, 6)            4614        ['dropout[0][0]']                
                                                                                                  
==================================================================================================
Total params: 109,486,855
Trainable params: 4,614
Non-trainable params: 109,482,241
```

The model consists of a BERT tokenizer, a BERT encoder, a dropout layer and a dense layer. The BERT tokenizer is used to tokenize the input text and the BERT encoder is used to encode the tokens into a vector representation. Here the pretrained [BERT base encoder](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4) from TensorFlow Hub is used alongwith the [BERT uncased tokenizer](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3).

The dropout layer is used to prevent overfitting and the dense layer is used to classify the encoded text into one of the six emotion categories.
## Libraries Used

The following libraries were used in this project:

- TensorFlow: To build sentiment classifier model.
- Gradio: To create the user interface for the sentiment classifier.

## License

This project is licensed under the [MIT License](LICENSE).