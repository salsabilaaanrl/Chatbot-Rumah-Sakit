 # -*- coding: utf-8 -*-
"""chatbot

# Telegram Chatbot Puskesmas Ngadirejo
"""

import pandas as pd
import numpy as np
import nltk
import string
import warnings
import requests
import random
from tokens import token
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

warnings.filterwarnings("ignore")

nltk.download('wordnet')
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer

"""## Data latih"""

data = [["Hai",0],["Halo",0],["Sore",0],["Assalamualaikum",0],["Salam",0],["Pagi",0],["Siang",0],["Malam",0],["Permisi",0],["hi",0],
        ["layanan apa saja yang tersedia hari ini?",1],["ada layanan apa saja hari ini?",1],["layanan apa yang tersedia?",1],["saya mau periksa hari ini",1],["apakah ada layanan gigi?",1],
        ["apakah hari ini bisa vaksin?",2],["bisa vaksin disini?",2], ["saya mau vaksin",2],["bisakah saya vaksin disini?",2],["vaksin apa saja yang tersedia?",2],   
        ["biasakah disini memakai bpjs?",3],["bpjs bisa dipakai disini?",3],["apakah bisa menggunakan bpjs",3],["apabisa pakai bpjs?",3],["puskesmas ini menerima bpjs?",3],
        [" ",4],[" ",4],[" ",4],[" ",4],[" ",4],[" ",4],[" ",4],[" ",4],[" ",4],[" ",4],
        ["Selamat tinggal",12],["Sampai jumpa",12],["Dadah",12],["Bye",12],["Sampai jumpa lagi",12],["Sampai jumpa lain waktu",12],
        ["Terima kasih",13],["Terimakasih",13],["Makasih",13],["Terima kasih banyak",13],["Makasih banyak",13],["Terima kasih banget",13],["Trims",13],["Makasi",13],["Terimakasi",13],
        ["Apa yang bisa kamu lakukan?",14],["Apa saja yang bisa kamu lakukan",14],["Apa yang anda lakukan?",14],["Apa saja yang anda lakukan?",14],
        ["Aku merasa sakit",15],["Saya merasa sakit",15],["Saya ingin berobat",15],["Aku ingin berobat",15],["Aku sakit",15],["Saya sakit",15]]

#Mengubah data menjadi DataFrame agar dapat dilatih
df = pd.DataFrame(data, columns = ["Text","Intent"])

"""## Text processing"""

# Lemmatisasi menggunakan WordNet

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

"""# Membuat dan melatih model"""

x = df['Text']
y = df['Intent']

#Deklarasi stopword
stop_factory = StopWordRemoverFactory()
stopwords = stop_factory.get_stop_words()
stopword = stop_factory.create_stop_word_remover()

vectorizer = TfidfVectorizer(tokenizer=Normalize, stop_words = stopwords)

X = vectorizer.fit_transform(x)

# Latih data dengan Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X, y)

# Test dengan kalimat
X_test = ["p"]
prediction = lr.predict(vectorizer.transform(X_test))

prediction

# Untuk mendapatkan probabilitas data uji yang ada di setiap Intent

a = lr.predict_proba(vectorizer.transform(X_test))
a

"""# Membuat respon untuk masing" intent"""

responses = {
      0 : {"intent":"salam","response":['Hai','Halo', 'Senang bertemu Anda','hi',]}, 
      1 : {"intent":"jadwal_pelayanan","response":['jadwal pelayanan di puskesmas ini:','pelayanan yang tersedia di puskesmas ini:','puskesmas ini melayani:',]},
      2 : {"intent":"info_vaksin","response":['vaksin yang tersedia di puskesmas ini:', 'di puskesmas ini ada vaksin:']},
      3 : {"intent":"bpjs","response":['disini bisa menggunakan layanan BPJS','di rumah sakit ini bisa menggunakan BPJS','layanan BPJS tersedia di rumah sakit ini']},
      4 : {"intent":"tidak_mengerti","response":['Maaf, saya tidak mengerti']},
      12 : {"intent":"selamat_tinggal","response":['Sampai jumpa','Selamat tinggal','dadah']},
      13 : {"intent":"terimakasih","response":['Sama-sama','Tidak masalah','Terima kasih kembali','Jangan sungkan']},
      14 : {"intent":"info","response":['Yang bisa saya lakukan, antara lain:']},
      15 : {"intent":"sakit","response":['segera hubungi dokter dan periksakan diri anda ke puskesmas']}}

"""# Mengaktifkan bot Telegram"""

import json

class telegram_bot():
    def __init__(self):
        self.token= token
        self.url = f"https://api.telegram.org/bot{self.token}"

    def get_updates(self,offset=None):
        url = self.url+"/getUpdates?timeout=100"
        if offset:
            url = url+f"&offset={offset+1}"
        url_info = requests.get(url)
        return json.loads(url_info.content)
    
    def send_message(self,msg,chat_id):
        url = self.url + f"/sendMessage?chat_id={chat_id}&text={msg}"
        if msg is not None:
            requests.get(url)

    def grab_token(self):
        return tokens

"""## Deploy bot dan membuat fungsi reply"""

def intent(user_response):
    text_intent = [user_response]
    X_test_intent = vectorizer.transform(text_intent)
    predicted_intent = lr.predict(X_test_intent)
    intent_predicted = responses[predicted_intent[0]]['intent']
    return intent_predicted

import random

def bot_initialize(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        
        user_intent = intent(user_response)
        
        if(user_intent != 'selamat_tinggal'):
            if(user_response == '/start'):
                resp = """Halo, saya Chatbot Puskesmas Ngadirejo. Saya harap saya dapat membantu Anda"""
                return resp

            elif (user_intent == 'info' or user_response == '/help'):
                resp = str(random.choice(responses[14]['response'])) +  '''
1. /start
2. /help
3. menyapa
4. Info Pelayanan
5. info Vaksin
6. info BPJS
7. ucapan selamat tinggal
8. balasan terima kasih
9. Error handling'''
                return resp
            
            elif (user_intent == 'salam'):
                resp = str(random.choice(responses[0]['response'])) + ", Ada yang bisa saya bantu?"
                return resp
        
            elif(user_intent == 'terimakasih'):
                resp = random.choice(responses[13]['response'])
                return resp
            
            elif(user_intent == 'jadwal_pelayanan'):
                resp = random.choice(responses[1]['response']) + '''
Pelayanan Pemeriksaan Umum
Pelayanan Kesehatan Gigi dan Mulut
Fisioterapi
Pelayanan KIA
Pelayanan VCT
Pelayanan KB
Imunisasi
Gizi
Sanitasi
MTBS'''
                return resp
            
            elif(user_intent == 'info_vaksin'):
                resp = random.choice(responses[2]['response']) + '''
Pfizer
Astra Zeneca
Sinovac'''
                return resp
            
            elif(user_intent == 'bpjs'):
                resp = "Ya, " + random.choice(responses[3]['response']) 
                return resp
           
            elif(user_intent == 'tidak_mengerti'):
                resp = random.choice(responses[4]['response'])
                return resp
            
            elif(user_intent == 'sakit'):
                resp = "Jika merasa sakit dan ingin berobat, " + random.choice(responses[15]['response']) + " https://goo.gl/maps/y6LhuGcaZQA56Xa99"
                return resp

            else:
                resp = "Maaf saya tidak mengerti, coba olah kata kembali"
                return resp
            
        else:
            flag = False
            resp = random.choice(responses[12]['response']) + ", semoga sehat selalu :)"
            return resp

tbot = telegram_bot()
update_id = None

def make_reply(msg):
    if msg is not None:
        reply = bot_initialize(msg)
    return reply

while True:
    print("siap menerima pesan")
    updates = tbot.get_updates(offset=update_id)
    updates = updates['result']
    print(updates)
    if updates:
        for item in updates:
            update_id = item["update_id"]
            print(update_id)
            try:
                message = item["message"]["text"]
                print(message)
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            print(from_)

            reply = make_reply(message)
            tbot.send_message(reply,from_)