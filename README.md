# Translator with Speech and GUI
This is a Neural Machine Translation (NMT) project that translates English sentences to French,Spanish using a Sequence-to-Sequence (Seq2Seq) model with LSTM layers. It includes a Tkinter GUI, text-to-speech (TTS), and speech-to-text (STT) functionalities for user interaction.

## 🚀 Features
<pre>
✅ Translates English to French using a trained LSTM model

🧑‍💻 GUI built with Tkinter

🧏‍♂️ Text-to-Speech for both English and French outputs using pyttsx3

🎤 Speech-to-Text input using speech_recognition and Google Speech API

📊 Displays training accuracy and confusion matrix

💾 Model is saved and reused from .h5 file

📦 Uses both English-Spanish and English-French parallel corpora (only 10 examples for demo)

</pre>

## 🛠️ Tech Stack
<pre>
Python

TensorFlow / Keras

NumPy, Pandas, Matplotlib, Seaborn

Tkinter

SpeechRecognition, pyttsx3, gTTS

</pre>
## 📁 Dataset
Used bilingual text files from:

eng_fra.txt – English to French

english to spanish.txt – Used for experimentation



## 🧠 Model Architecture
### Encoder:

2 stacked LSTM layers

Input: One-hot encoded English characters

### Decoder:

2 stacked LSTM layers

Dense layer with softmax activation for output characters

Teacher forcing used during training

Loss Function: Categorical Crossentropy

Optimizer: RMSprop or Adam


## 🖥️ GUI Features (Tkinter)
<pre>
Input and output text boxes

### Buttons:

Translate: Translates English to French

🔊 (English/French): Reads the input/output aloud

🎤: Converts speech to English text

Reset: Clears the input box

</pre>


## 🧪 How to Run

### 1. Install Requirements

pip install numpy pandas keras tensorflow matplotlib seaborn pyttsx3 speechrecognition gTTS

### 2. Download Dataset
Place eng_fra.txt and english to spanish.txt in your working directory.

### 3. Run the Program
python code.py

## 📈 Evaluation
Shows accuracy and confusion matrix using matplotlib and seaborn.



