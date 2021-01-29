import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import os
# nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_word(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)

    for index, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[index] = 1.0
    return bag

def activate_mic():
    global frase
    microphone = sr.Recognizer()

    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source)
        # print('Microfone...')
        audio = microphone.listen(source)
    try:
        frase = microphone.recognize_google(audio, language='pt-BR')
        print('Usuário: ', frase)
    except sr.UnknownValueError:
        print('Mina: Isso não funcionou!')

    return frase

def create_audio(text):
    tts = gTTS(text, lang='pt-BR')
    tts.save('C:/Users/mjuni/OneDrive/Documentos/Junior/Chatbot/bot3.mp3')
    playsound('C:/Users/mjuni/OneDrive/Documentos/Junior/Chatbot/bot3.mp3')
    os.remove('C:/Users/mjuni/OneDrive/Documentos/Junior/Chatbot/bot3.mp3')
