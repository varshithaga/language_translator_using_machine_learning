        from sklearn.metrics import confusion_matrix, accuracy_score
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tkinter import ttk,messagebox
        from tkinter import ttk, Text, END, GROOVE, Frame, Label, Button
        import pyttsx3
        import speech_recognition as sr
        from tkinter import messagebox
        import tkinter as tk
        from gtts import gTTS
        import numpy as np 
        import pandas as pd
        from keras.layers import Dense
        from keras.layers import LSTM,Input
        from keras.models import Model
        import keras
        from tensorflow.keras.models import load_model
        
        
        english_texts=[]
        german_texts=[]
        english_character=[]
        german_character=[]
        with open(r"C:\Users\hemuh\Downloads\archive (2)\english to spanish.txt", "r", encoding="utf-8") as f:
        with open(r"C:\Users\hemuh\Downloads\archive (3)\eng_fra.txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        for line in lines[:10000]:
            english_text,german_text=line.split("\t")
            english_texts.append(english_text)
            german_text = "\t" + german_text + "\n"
            german_texts.append(german_text)
        
        for i in english_texts:
            for c in i:
                if c not in english_character:
                    english_character.append(c)
                    english_character.sort()
        
        for j in german_texts:
            for c in j:
                if c not in german_character:
                    german_character.append(c)
                    german_character.sort()
        
        
        english_d={}
        for i in range(len(english_character)):
            english_d[english_character[i]]=i
        
        german_d={}
        for i in range(len(german_character)):
            german_d[german_character[i]]=i
        
        english_encoder_tokens = len(english_character)
        german_decoder_tokens = len(german_character)
        
        
        max_encoder_seq_length=0
        for i in english_texts:
            if len(i)>max_encoder_seq_length:
                max_encoder_seq_length=len(i)
        
        
        max_decoder_seq_length=0
        for i in german_texts:
            if len(i)>max_decoder_seq_length:
                max_decoder_seq_length=len(i)
        
        
        encoder_input_data=[]
        for bb in range(10):
            a=[]
            b=[]
            c=[]
            k=len(english_texts[bb])
            m=0
            while m<k:
                for char in english_texts[bb][m]:
                                   for i in range(len(english_character)):
                                                    if english_d[char]==i:
                                                        a.append(1)
                                                    else:
                                                        a.append(0)
               
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            while m<max_encoder_seq_length:
                for i in range(len(english_character)):
                    if i==0:
                        a.append(1)
                    else:
                        a.append(0)
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            encoder_input_data.append(c)
        
        
        encoder_input_data=np.array(encoder_input_data)
        
        
        decoder_input_data=[]
        for bb in range(10):
            a=[]
            b=[]
            c=[]
            k=len(german_texts[bb])
            m=0
            while m<k:
                for char in german_texts[bb][m]:
                                   for i in range(len(german_character)):
                                                    if german_d[char]==i:
                                                        a.append(1)
                                                    else:
                                                        a.append(0)
               
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            while m<max_decoder_seq_length:
                for i in range(len(german_character)):
                    if i==0:
                        a.append(1)
                    else:
                        a.append(0)
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            decoder_input_data.append(c)
        
        
        decoder_input_data=np.array(decoder_input_data)
        
        
        decoder_target_data=[]
        for bb in range(10):
            a=[]
            b=[]
            c=[]
            k=len(german_texts[bb])
            m=1
            while m<k:
                for char in german_texts[bb][m]:
                                   for i in range(len(german_character)):
                                                    if german_d[char]==i:
                                                        a.append(1)
                                                    else:
                                                        a.append(0)
               
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            m=m-1
            while m<max_decoder_seq_length:
                for i in range(len(german_character)):
                    if i==0:
                        a.append(1)
                    else:
                        a.append(0)
                for kp in a:
                    b.append(kp)
                c.append(b)
                b=[]
                a=[]
                m=m+1
            decoder_target_data.append(c)
        
        decoder_target_data=np.array(decoder_target_data)
        
        batch_size = 64   Batch size for training.
        epochs = 100  Number of epochs to train for.
        latent_dim = 256
        
        
        encoder_inputs = Input(shape=(None,len(english_character)))
        encoder = LSTM(latent_dim,dropout=0.2,return_sequences=True,return_state=True)
        encoder_outputs_1, state_h_1, state_c_1 = encoder(encoder_inputs)
        encoder = LSTM(latent_dim,dropout=0.2,return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_outputs_1)
        encoder_states = [state_h_1,state_c_1,state_h, state_c]
        
        
        decoder_inputs = Input(shape=(None, len(german_character)))
        decoder_lstm = LSTM(latent_dim,return_sequences=True,dropout=0.2,return_state=True)
        decoder_outputs_1, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h_1,state_c_1])
        decoder_lstm_1 = LSTM(latent_dim, return_sequences=True,dropout=0.2,return_state=True)
        decoder_outputs, _, _ = decoder_lstm_1(decoder_outputs_1, initial_state=[state_h,state_c])
        decoder_dense = Dense(len(german_character), activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)
        model=Model([encoder_inputs, decoder_inputs], decoder_outputs) 
        model.summary()
        
        
        model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
        )
        
        model.save('englishtospanish.h5')
        from tensorflow.keras.models import load_model
        model=load_model('englishtospanish.h5')
        model = keras.models.load_model('englishtospanish.h5')

        model.save('englishtofrench.h5')
        from tensorflow.keras.models import load_model
        model=load_model('englishtofrench.h5')
        model = keras.models.load_model("englishtofrench.h5")

        model = keras.models.load_model("englishtofrench.h5")
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        encoder_inputs = model.input[0]   input_1
        encoder_outputs_1, state_h_enc_1, state_c_enc_1 = model.layers[2].output 
        encoder_outputs, state_h_enc, state_c_enc = model.layers[4].output 
        encoder_states = [state_h_enc_1, state_c_enc_1,state_h_enc, state_c_enc]
        encoder_model_1 = keras.Model(encoder_inputs, encoder_states)
        
        decoder_inputs = model.input[1]  
        decoder_state_input_h = keras.Input(shape=(latent_dim,),)
        decoder_state_input_c = keras.Input(shape=(latent_dim,),)
        decoder_state_input_h1 = Input(shape=(latent_dim,),)
        decoder_state_input_c1 = Input(shape=(latent_dim,),)
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,decoder_state_input_h1,decoder_state_input_c1]
        decoder_lstm = model.layers[3]
        dec_o, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs[:2])
        decoder_lstm_1=model.layers[5]
        dec_o_1, state_h1, state_c1 = decoder_lstm_1(
            dec_o, initial_state=decoder_states_inputs[-2:])
        decoder_states = [state_h,state_c,state_h1,state_c1]
        decoder_dense = model.layers[6]
        decoder_outputs = decoder_dense(dec_o_1)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        
        reverse_input_char_index ={}
        for i in range(len(english_character)):
            reverse_input_char_index[i]=english_character[i]
        reverse_target_char_index ={}
        for i in range(len(german_character)):
            reverse_target_char_index[i]=german_character[i]
        
        def decode_sequence(input_seq):
            states_value=encoder_model_1.predict(input_seq)
            target_seq = np.zeros((1, 1, len(german_character)))
            target_seq[0, 0, german_d["\t"]] = 1.0
            flag=0
            sent=""
            while flag==0:
                output_tokens, h, c,h1,c1 = decoder_model.predict([target_seq] + states_value)
                sample = np.argmax(output_tokens[0, -1, :])
                sampled_char = reverse_target_char_index[sample]
                sent+=sampled_char
                if sampled_char == "\n" or len(sent) > max_decoder_seq_length:
                    flag=1
                target_seq = np.zeros((1, 1, len(german_character)))
                target_seq[0, 0,sample] = 1.0
                states_value = [h, c,h1,c1]
            return sent
        

        def translate_text():
            english = text1.get("1.0", END).strip()
            print("You entered:", english)
        
             Prepare the input for the model
            k = len(english)
            m = 0
            a = []
            b = []
            c = []
            inpu = []
        
            while m < k:
                for char in english[m]:
                    for i in range(len(english_character)):
                        if english_d[char] == i:
                            a.append(1)
                        else:
                            a.append(0)
                for kp in a:
                    b.append(kp)
                c.append(b)
                b = []
                a = []
                m = m + 1
            while m < max_encoder_seq_length:
                for i in range(len(english_character)):
                    if i == 0:
                        a.append(1)
                    else:
                        a.append(0)
                for kp in a:
                    b.append(kp)
                c.append(b)
                b = []
                a = []
                m = m + 1
            inpu.append(c)
        
            inpu = np.array(inpu)
            inpu.shape
        
            d = decode_sequence(inpu)
        
            text2.delete("1.0", END)
            text2.insert(END, d)
        
            return d
        
        history = model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
        )
        
         
        y_true = np.argmax(decoder_target_data, axis=-1).flatten()
        y_pred = np.argmax(model.predict([encoder_input_data, decoder_input_data]), axis=-1).flatten()
        
        
        cm = confusion_matrix(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
        
        print(f"Accuracy: {acc}")

         def speak_text2():
             translated_text = text2.get("1.0", END).strip()
             engine = pyttsx3.init()
             engine.say(translated_text)
             engine.runAndWait()
        
        def speak_text1():
            inputed_text = text1.get("1.0", END).strip()
            engine = pyttsx3.init()
            engine.say(inputed_text)
            engine.runAndWait()
        def speak_text2():
            inputed_text = text2.get("1.0", END).strip()
            engine = pyttsx3.init()
            engine.say(inputed_text)
            engine.runAndWait()
        
        def convert_speech_to_text():
            r = sr.Recognizer()
            with sr.Microphone () as source:
                try:
                    audio=r.listen (source)
                    text=r.recognize_google(audio)
                    text1.insert(tk.END, text + "\n")
                except sr.UnknownValueError:
                    text1.insert(tk.END, "Could not understand audio\n")
                except sr.RequestError as e:
                    text1.insert(tk.END, "Error: {0}\n".format(e))

                    
        
        def reset_text_widget():
            text1.delete("1.0", tk.END)
            
        root = tk.Tk()
        root.title("Translator")
        root.geometry("1080x400")
        root.resizable(False, False)
        root.configure(background="white")
        
        label1 = Label(root, text="English", font="segoe 30 bold", bg="white", width=12, bd=5, relief=GROOVE)
        label1.place(x=10, y=50)
        
        label2 = Label(root, text="French", font="segoe 30 bold", bg="white", width=12, bd=5, relief=GROOVE)
        label2.place(x=620, y=50)
        
        f = Frame(root, bg="Black", bd=5)
        f.place(x=10, y=118, width=440, height=210)
        
        text1 = Text(f, font="Roboto 20", bg="white", relief=GROOVE)
        text1.place(x=0, y=0, width=430, height=200)
        
        f1 = Frame(root, bg="Black", bd=5)
        f1.place(x=620, y=118, width=440, height=210)
        
        text2 = Text(f1, font="Roboto 20", bg="white", relief=GROOVE)
        text2.place(x=0, y=0, width=430, height=200)
        
        translate = Button(root, text="Translate", font="Roboto 15", activebackground="white", cursor="hand2", bd=1, width=10, height=2, bg="black", fg="white", command=translate_text)
        translate.place(x=476, y=250)
        
        speak2 = Button(root, text="🔊", font="Roboto 15", activebackground="white", cursor="hand2", bd=1, width=10, height=1, bg="black", fg="white", command=speak_text2)
        speak2.place(x=800, y=340)
        
        speak1 = Button(root, text="🔊", font="Roboto 15", activebackground="white", cursor="hand2", bd=1, width=10, height=1, bg="black", fg="white", command=speak_text1)
        speak1.place(x=310, y=340)
        
        
        convert = Button(root, text="🎤", font="Roboto 15", activebackground="white", cursor="hand2", bd=1, width=10, height=1, bg="black", fg="white", command=convert_speech_to_text)
        convert.place(x=30, y=340)
        
        reset = Button(root, text="Reset", font="Roboto 15", activebackground="white", cursor="hand2", bd=1, width=10, height=1, bg="black", fg="white", command=reset_text_widget)
        reset.place(x=170, y=340)
        
        root.mainloop()