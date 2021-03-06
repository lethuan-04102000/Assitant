from kivymd.uix import dialog
from kivymd.uix import button
from kivymd.uix.screen import MDScreen
from kivymd.app import MDApp
from kivymd.uix.list import MDList, OneLineListItem, TwoLineListItem
from kivy.uix.image import Image
from kivymd.uix.button import MDFillRoundFlatIconButton, MDFillRoundFlatButton, MDFlatButton, MDRaisedButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.label import MDLabel
from kivymd.uix.toolbar import MDToolbar
from kivymd.uix.dialog import MDDialog
from kivy.lang import Builder
from kivy.factory import Factory
from kivymd.app import MDApp
from kivy.uix.scrollview import ScrollView
from googletrans import Translator
import os
import bs4
import pyautogui
import playsound
from kivy.core.window import Window
from win10toast import ToastNotifier
import speech_recognition as sr
import time
import sys
import ctypes
import wikipedia
import datetime
import json
import re
import webbrowser
import numpy as np
import tensorflow as tf
import tflearn
import random
import json
import pickle
import smtplib
import requests
import urllib
import urllib.request as urllib2
from time import strftime
from gtts import gTTS
from youtube_search import YoutubeSearch
from bs4 import BeautifulSoup
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
# load model and data
data = pickle.load(open("training_data", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.load('Model/model.tflearn')
with open('Data/intents.json', encoding='utf8') as json_data:
    intents = json.load(json_data)


def clean_up_sentence(sentence):
    # It Tokenize or Break it into the constituents parts of Sentense.
    sentence_words = nltk.word_tokenize(sentence)
    # Stemming means to find the root of the word.
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Return the Array of Bag of Words: True or False and 0 or 1 for each word of bag that exists in the Sentence
def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

ERROR_THRESHOLD = 0.25
print("ERROR_THRESHOLD = 0.25")
def classify(sentence):
    # Prediction or To Get the Posibility or Probability from the Model
    results = model.predict([bow(sentence, words)])[0]
    # Exclude those results which are Below Threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # Sorting is Done because heigher Confidence Answer comes first.
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        # Tuppl -> Intent and Probability
        return_list.append((classes[r[0]], r[1]))
    return return_list


def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # That Means if Classification is Done then Find the Matching Tag.
    if results:
        # Long Loop to get the Result.
        while results:
            for i in intents['intents']:
                if results[0][1] > 0.8:
                    if i['tag'] == results[0][0]:
                        # Random Response from High Order Probabilities
                        return random.choice(i['responses'])
                # Tag Finding
            results.pop(0)
name = 'Nam'
language = 'vi'
toast = ToastNotifier()

alone = 'C??u n??y h???i kh?? qu?? ??i, Ng?????i y??u ????u ph???i mu???n th?? c?? lu??n. K??? ra l???i th???y th??m bu???n, B???n l??m tr??? l?? alone d??i d??i haha'
like = 'T??i th??ch t??m ki???m th??ng tin v?? gi??p ????? m???i ng?????i. Th?? c?? t??nh l?? y??u kh??ng'


def current_weather(text):
    reg_ex = re.search('??? (.+)', text)
    domain = reg_ex.group(1)
    ow_url = "http://api.openweathermap.org/data/2.5/weather?"
    city = domain
    if not city:
        pass
    api_key = "fe8d8c65cf345889139d8e545f57819a"
    call_url = ow_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    response = requests.get(call_url)
    data = response.json()
    if data["cod"] != "404":
        city_res = data["main"]
        current_temperature = city_res["temp"]
        current_pressure = city_res["pressure"]
        current_humidity = city_res["humidity"]
        suntime = data["sys"]
        sunrise = datetime.datetime.fromtimestamp(suntime["sunrise"])
        sunset = datetime.datetime.fromtimestamp(suntime["sunset"])
        wthr = data["weather"]
        weather_description = wthr[0]["description"]
        now = datetime.datetime.now()
        content = """
        H??m nay l?? ng??y {day} th??ng {month} n??m {year}
        M???t tr???i m???c v??o {hourrise} gi??? {minrise} ph??t
        M???t tr???i l???n v??o {hourset} gi??? {minset} ph??t
        Nhi???t ????? trung b??nh l?? {temp} ????? C
        ??p su???t kh??ng kh?? l?? {pressure} h??c t?? Pascal
        ????? ???m l?? {humidity}%
        Tr???i h??m nay quang m??y. D??? b??o m??a r???i r??c ??? m???t s??? n??i.""".format(day=now.day, month=now.month, year=now.year, hourrise=sunrise.hour, minrise=sunrise.minute,
                                                                           hourset=sunset.hour, minset=sunset.minute,
                                                                           temp=current_temperature, pressure=current_pressure, humidity=current_humidity)
        speak(content)
        return(content)
        time.sleep(20)
    else:
        speak("Kh??ng t??m th???y ?????a ch??? c???a b???n")


def covid_api(text):
    text = text.lower()
    if 't???i' in text:
        reg_ex = re.search('t???i (.+)', text)
    elif '???' in text:
        reg_ex = re.search('??? (.+)', text)
    if len(text) > 0:
        if 'anh' in text:
            country = 'uk'
        elif 'm???' in text or 'hoa k???' in text or 'usa' in text:
            country = 'us'
        elif 'vi???t' in text or 'vi???t nam' in text or 'viet' in text:
            country = 'viet-nam'
        else:
            translator = Translator()
            translated = translator.translate('n?????c '+text, dest='en')
            country = translated.text
        country = country.lower()
        try:
            url = 'https://www.worldometers.info/coronavirus/country/' + country + '/'
            html_data = data = requests.get(url)
            bs = bs4.BeautifulSoup(html_data.text, 'html.parser')
            data = []
            data_ = ''
            info_div = bs.find(
                'div', class_='content-inner').findAll('div', id='maincounter-wrap')
            for s in info_div:
                s = s.find("span", class_=None)
                clean = re.compile('<.*?>')
                s = re.sub(clean, '', str(s))
                s = s.replace(',', '.')
                data.append(s)
            header = ['S??? ca m???c: ', 'Ca t??? vong: ', 'Ca ch???a kh???i: ']
            for i in range(len(header)):
                data_ = data_+header[i]+data[i]+', \n'
            speak(data_)
            return(data_)
        except:
            url = 'https://www.worldometers.info/coronavirus/'
            html_data = data = requests.get(url)
            bs = bs4.BeautifulSoup(html_data.text, 'html.parser')
            data = []
            data_ = ''
            info_div = bs.find(
                'div', class_='content-inner').findAll('div', id='maincounter-wrap')
            for s in info_div:
                s = s.find("span", class_=None)
                clean = re.compile('<.*?>')
                s = re.sub(clean, '', str(s))
                s = s.replace(',', '.')
                data.append(s)
            header = ['S??? ca m???c: ', 'Ca t??? vong: ', 'Ca ch???a kh???i: ']
            for i in range(len(header)):
                data_ = data_+header[i]+data[i]+', \n'
            speak(data_)
            return(data_)
    else:
        s = 'Kh??ng c?? k???t qu???'
        speak(s)
        return s


def get_text():
    for i in range(3):
        print('N??i ??i')
        text = get_voice()
        if(text):
            return text.lower()
        elif i < 2:
            speak("T??i kh??ng nghe r??, b???n c?? th??? n??i l???i kh??ng")
    time.sleep(10)
    stop()
    return 0


def talk(name):
    day_time = int(strftime("%H"))
    if day_time < 12:
        hi = "Ch??o bu???i s??ng {}. Ch??c b???n ng??y m???i t???t l??nh!".format(name)
        speak(hi)
    elif day_time < 18:
        hi = "Ch??o bu???i chi???u {}".format(name)
        speak(hi)
    else:
        hi = "Ch??o bu???i t???i {}".format(name)
        speak(hi)
    time.sleep(5)
    speak("B???n c?? kh???e kh??ng ?")
    time.sleep(3)
    answer = get_voice()
    if answer:
        if "c??" in answer:
            speak("Th???t l?? t???t")
        else:
            speak("Bu???n qu??, v???y b???n n??n ngh??? ng??i cho kh???e ??i!")


def get_name():
    speak("T??n b???n l?? {}".format(name))
    return "T??n b???n l?? {}".format(name)


def open_application(text):
    if "microsoft edge" in text:
        speak("M??? Microsoft Edge")
        os.startfile(
            'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe')
        time.sleep(3)
        speak("???? M??? Microsoft Edge")
        time.sleep(3)
        return '???? M??? Microsoft Edge'

    elif "google" in text:
        speak("M??? Google Chrome")
        os.startfile('C:\Program Files\Google\Chrome\Application\chrome.exe')
        time.sleep(3)
        speak("???? m??? Google Chrome")
        time.sleep(3)
        return '???? m??? Google Chrome'

    elif "font" in text:
        speak("M??? Font Lab")
        os.startfile('C:\Program Files\Fontlab\FontLab 7\FontLab 7.exe')
        time.sleep(3)
        speak("???? m??? Font Lab")
        time.sleep(3)
        return '???? m??? FontLab'
    elif "word" in text:
        speak("M??? Microsoft Word")
        os.startfile(
            'C:\Program Files\Microsoft Office\\root\Office16\WINWORD.EXE')
        time.sleep(3)
        speak("???? m??? Microsoft Word")
        time.sleep(3)
        return("???? m??? Microsoft Word")
    elif "excel" in text:
        speak("M??? Microsoft Excel")
        os.startfile(
            'C:\Program Files\Microsoft Office\\root\Office16\EXCEL.EXE')
        time.sleep(3)
        speak("???? m??? Microsoft Excel")
        time.sleep(3)
        return '???? m??? Microsoft Excel'
    elif "c??i ?????t" in text or 'setting' in text:
        speak("M??? C??i ?????t")
        os.popen("start ms-settings:")
        time.sleep(3)
        speak("???? m??? C??i ?????t")
        time.sleep(3)
        return '???? m??? C??i ?????t'
    elif "??i???u khi???n" in text or 'Control' in text:
        speak("M??? Tr??nh ??i???u khi???n")
        os.popen("control panel")
        time.sleep(3)
        speak("???? m??? tr??nh ??i???u khi???n")
        time.sleep(3)
        return '???? m??? Tr??nh ??i???u khi???n'
    else:
        speak("???ng d???ng ch??a ???????c c??i ?????t. B???n h??y th??? l???i!")
        time.sleep(3)
        return '???ng d???ng ch??a ???????c c??i ?????t. B???n h??y th??? l???i!'


def help():
    time.sleep(3)
    speak("B???n c???n T??i gi??p g?? kh??ng?")
    time.sleep(3)
    return'B???n c???n T??i gi??p g?? kh??ng?' 


def call_bot(text):
    text = text.lower()
    answer = response(text)
    answer = str(answer)
    if answer == "1":
        s = 'Ch??o b???n '+name + ' t??i c?? th??? gi??p g?? cho b???n'
        speak(s)
        return s
    elif answer == "2":
        s = "H???n g???p l???i b???n nh??"
        speak(s)
        return s
    elif answer == "3":
        s = 'Ng???i qu??, T??i r???t vui khi gi??p ???????c b???n'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "4":
        s = 'T??i l??c n??o c??ng kh???e, tr??? khi m???t ??i???n m?? th??i h?? h??'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "5":
        s = 'T??n t??i l?? tr??? l?? VKU'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "6":
        s = 'Th??? gi???i r???t c?? nhi???u ng?????i nh??ng b???n lu??n l?? nh???t !'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "7":
        s = 'Bu???n qu??, b???n n??n ngh??? ng???i ??i nh?? !'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "8":
        s = 'T??i ???????c t???o ra b???i nh??m NC si??u c???p vip pro'
        speak(s)
        time.sleep(3)
        return s
    elif answer == "9":
        s = 'B???n t??n l?? ' + name
        speak(s)
        time.sleep(3)
        return s
    elif "covid t???i" in text or "corona t???i" in text or 'covid19 t???i' in text or 's??? ca covid' in text:
        return covid_api(text)
    elif "tr?? chuy???n" in text or "n??i chuy???n" in text:
        return talk(name)
    elif "??m l?????ng" in text:
        return change_volume(text)
    elif "th???i ti???t ???" in text:
        return current_weather(text)
    elif "???ng d???ng" in text:
        return open_application(text)
    elif "t??n t??i" in text or "t??i t??n" in text:
        return get_name()
    elif "b???n c?? ng?????i y??u ch??a" in text or "ng?????i y??u c???a b???n" in text:
        speak(alone)
        time.sleep(5)
        return alone
    elif "b???n c?? kh???e kh??ng" in text or "kh???e kh??ng" in text:
        speak("T??i lu??n lu??n kh???e tr??? khi m???t ??i???n")
        time.sleep(4)
        return 'T??i lu??n lu??n kh???e tr??? khi m???t ??i???n'
    elif "trang web" in text:
        return open_website(text)
        time.sleep(4)
    elif "gi?? v??ng" in text:
        s = gold_price()
        speak(s)
        # toast.show_toast("Gi?? v??ng",s)
        return s
    elif "t??m ki???m" in text:
        return open_website_search(text)
        time.sleep(4)
    elif "hi???n t???i" in text:
        s = get_time(text)
        # toast.show_toast("Gi???",s)
        return s
    elif "?????nh ngh??a" in text:
        return speak("Bot: haha")
    elif "ch??i nh???c" in text:
        return play_song(text)
    else:
         return query(text)


def search(text):
    url = "https://www.google.com.tr/search?q={}".format(text)
    webbrowser.open(url)
    s = "Kh??ng c?? th??ng tin, v?? sau ????y l?? 1 v??i k???t qu???"
    speak(s)
    return s


def open_website_search(text):
    reg_ex = re.search('ki???m (.+)', text)
    if reg_ex:
        domain = reg_ex.group(1)
        url = "https://www.google.com.tr/search?q={}".format(domain)
        webbrowser.open(url)
        speak("Trang web b???n y??u c???u ???? ???????c m???.")
        return True
    else:
        return False


def get_time(text):
    now = datetime.datetime.now()
    if "gi???" in text:
        speak('B??y gi??? l?? %d gi??? %d ph??t' % (now.hour, now.minute))
        return('B??y gi??? l?? %d gi??? %d ph??t' % (now.hour, now.minute))
    elif "ng??y" in text:
        speak("H??m nay l?? ng??y %d th??ng %d n??m %d" %
              (now.day, now.month, now.year))
        return("H??m nay l?? ng??y %d th??ng %d n??m %d" %
               (now.day, now.month, now.year))
    else:
        speak("Bot ch??a hi???u ?? c???a b???n. B???n n??i l???i ???????c kh??ng?")


def open_website(text):
    reg_ex = re.search('web (.+)', text)
    if reg_ex:
        domain = reg_ex.group(1)
        url = 'https://www.' + domain
        webbrowser.open(url)
        speak("Trang web b???n y??u c???u ???? ???????c m???.")
        return 'Trang web b???n y??u c???u ???? ???????c m???.'
    else:
        return False


def stop():
    speak("H???n g???p l???i b???n nh??")
    return 'H???n g???p l???i b???n nh??'


def language_voice(text):
    reg_ex = re.search('ti???ng (.+)', text)
    if reg_ex:
        if reg_ex:
            domain = reg_ex.group(1)
            if 'vi???t' in domain:
                return 'vi'
            if 'nh???t' in domain:
                return 'ja'
            if 'anh' in domain:
                return 'en'
            if 'trung' in domain:
                return 'zh'
            if 'ph??p' in domain:
                return 'fr'
            if 'h??n' in domain:
                return 'ko'
            else:
                return 'vi'
    else:
        return 'vi'


def gold_price():
    url = 'https://www.pnj.com.vn/blog/gia-vang/'
    page = urllib.request.urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    sult = soup.find(class_='tabBody mgbt15').find_all("td")
    CLEANR = re.compile('<.*?>')
    a2 = []
    for a in sult:
        c = re.sub(CLEANR, '', str(a))
        c = c.replace(',', '.')
        a2.append(c)
    a1 = 'Gi?? b??n v??ng ' + a2[0]+': ' + a2[1] + ' ?????ng' + '\n'+'Gi?? mua v??ng ' + a2[0]+': ' + a2[2] + \
        ' ?????ng'+'\n'+'Gi?? b??n v??ng ' + \
        a2[3]+': ' + a2[4] + ' ?????ng'+'\n'+'Gi?? mua v??ng ' + \
        a2[3]+': ' + a2[5] + ' ?????ng'
    return a1


def get_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("T??i: ", end='')
        audio = r.listen(source, phrase_time_limit=5)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return 0


def Toast(name, text):
    toast.show_toast(name, text)


def play_song(text):
    reg_ex = re.search('b??i (.+)', text)
    domain = reg_ex.group(1)
    mysong = domain
    while True:
        result = YoutubeSearch(mysong, max_results=10).to_dict()
        if result:
            break
    url = 'https://www.youtube.com' + result[0]['url_suffix']
    webbrowser.open(url)
    speak("B??i h??t b???n y??u c???u ???? ???????c m???.")
    return 'B??i h??t b???n y??u c???u ???? ???????c m??? ' + url


def speak2(lang, text):
    language_v = language_voice(lang)
    print("Bot: {}".format(text))
    # truyen vao text de doc language
    tts = gTTS(text=text, lang=language_v, slow=False)
    # luu am thanh vao he thong
    tts.save('sound.mp3')
    # play song truyen tu text
    playsound.playsound('sound.mp3', False)
    # xoa song
    os.remove('sound.mp3')


def query(text):
    user_query = text
    URL = "https://www.google.co.in/search?q=" + user_query
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
    }
    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    bebe = str(soup)
    # d???ch
    if 'class="Y2IQFc"' in bebe:
        sult = soup.find(id='tw-target-text').get_text()
        speak2(user_query, sult)
        return sult
    elif 'class="vk_bk dDoNo FzvWSb"' in bebe:
        sult = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
        speak(sult)
        time.sleep(6)
        return sult
    # th??ng tin c?? b???n
    elif 'class="Z0LcW"' in bebe:
        sult = soup.find(class_='Z0LcW').get_text()
        speak(sult)
        return sult
    # gi?? bicoin
    elif 'class="pclqee"' in bebe:
        sult = soup.find(class_='pclqee').get_text()
        speak(sult+'VN??')
        return sult+' VN??'
    elif 'class="LGOjhe"' in bebe:
        sult = soup.find(class_='LGOjhe').find(class_='hgKElc').get_text()
        speak(sult)
        return sult
    # lyric
    elif 'class="Oh5wg"' in bebe:
        sult = soup.find(class_='Oh5wg').find_all('span', jsname="YS01Ge")
        a = ''
        for lyric in sult:
            clean = re.compile('<.*?>')
            lyric = re.sub(clean, '', str(lyric))
            a += lyric+'\n'
        f = open('data.txt', 'w', encoding="utf8")
        f.write(a)
        s = 'L???i b??i h??t ???? ???????c m???'
        speak(s)
        os.startfile('data.txt')
        return(s)
    elif 'class="FzvWSb"' in bebe:
        sult = soup.find(class_='FzvWSb').get_text()
        speak(sult)
        return sult
    # m??y t??nh
    elif 'class="z7BZJb XSNERd"' in bebe:
        sult = soup.find(class_='qv3Wpe').get_text()
        speak(text + ' l?? ' + sult)
        return sult
    # khoang c??ch
    elif 'class="BbbuR uc9Qxb uE1RRc"' in bebe:
        sult = soup.find(class_='BbbuR uc9Qxb uE1RRc').find_all(
            'span', class_=None)
        a = ''
        for lyric in sult:
            clean = re.compile('<.*?>')
            lyric = re.sub(clean, '', str(lyric))
            a += lyric+' '
            a = a.replace(',', '.')
        speak(a)
        return a
    # ti???n t???
    elif 'class="dDoNo ikb4Bb gsrt"' in bebe:
        sult = soup.find(class_='dDoNo ikb4Bb gsrt').get_text()
        speak(text + ' l?? ' + sult)
        return sult
    # c??o b??n ph???i wikipedia
    elif 'class="kno-rdesc"' in bebe:
        sult = soup.find(class_='kno-rdesc').find('span')
        CLEANR = re.compile('<.*?>')
        sult = re.sub(CLEANR, '', str(sult))
        speak(sult)
        return sult
    elif 'class="ayRjaf"' in bebe:
        sult = soup.find(class_='zCubwf').get_text()
        speak(sult)
        return sult
    # ?????i c??c ????n v???
    elif 'class="dDoNo vrBOv vk_bk"' in bebe:
        sult = soup.find(class_='dDoNo vrBOv vk_bk').get_text()
        speak(sult)
        return sult
    # descript
    elif 'class="hgKElc"' in bebe:
        sult = soup.find(class_='hgKElc').get_text()
        speak(sult)
        return sult
    # thoi tiet
    elif 'class="UQt4rd"' in bebe:
        nhietdo = 'Nhi???t ?????: ' + \
            soup.find(class_='q8U8x').get_text() + '??C.'+'\n'
        doam = '????? ???m: ' + soup.find(id='wob_hm').get_text()
        mua = 'Kh??? n??ng c?? m??a: ' + soup.find(id='wob_pp').get_text()+'\n'
        gdp = soup.find(class_='wob_tci')
        wheather = gdp['alt']+'\n'
        nam = wheather + nhietdo + mua + doam
        # toast.show_toast("Th???i ti???t",nam)
        speak(nam)
        return nam
    elif 'class="gsrt vk_bk FzvWSb YwPhnf"' in bebe:
        sult = soup.find(class_='gsrt vk_bk FzvWSb YwPhnf').get_text()
        speak(sult)
        return sult
    else:
        if len(text) > 0:
            return search(text)


def change_volume(text):
    if 'gi???m' in text:
        try:
            s = re.findall(r'\d', text)
            s = int(''.join(s))
        except:
            s = 1
        pyautogui.press('volumedown', presses=s)
    if 't??ng' in text:
        try:
            s = re.findall(r'\d', text)
            s = int(''.join(s))
        except:
            s = 1
        pyautogui.press('volumeup', presses=s)
    if 't???t' in text:
        pyautogui.press('volumemute')
    speak('???? ch???nh')
    return '???? ch???nh th??nh c??ng'


def speak(text):
    print("Bot: {}".format(text))
    # truyen vao text de doc language
    tts = gTTS(text=text, lang=language, slow=False)
    # luu am thanh vao he thong
    tts.save('sound.mp3')
    # play song truyen tu text
    playsound.playsound('sound.mp3', False)
    # xoa song
    os.remove('sound.mp3')


class ConverterApp(MDApp):
    def show_data(self, txt):
        dialog = MDDialog(text=txt)
        dialog.open()

    def text(self, text, ):
        self.item = TwoLineListItem(
            text="B???n",
            secondary_text=self.input.text,
            height=25)
        self.list_view.add_widget(self.item)
        self.scroll.scroll_to(self.item)
        text = self.input.text
        text = text.lower()
        s = call_bot(text)
        if s:
            self.item = TwoLineListItem(
                text='Bot',
                secondary_text=s,)
            self.list_view.add_widget(self.item)
            self.scroll.scroll_to(self.item)
            self.close_button = MDRaisedButton(
                text='????ng', on_release=self.close_dialog)
            self.dialog = MDDialog(
                title='K???t qu???', text=s, buttons=[self.close_button])
            self.dialog.open()

    def close_dialog(self, obj):
        self.dialog.dismiss()

    def voice(self, text, ):
        speak('B???n c???n t??i gi??p g?? n??o')
        text = get_text()
        text = text.lower()
        if text:
            self.item = TwoLineListItem(
                text="B???n",
                secondary_text=text,
                height=25)
            self.list_view.add_widget(self.item)
            self.scroll.scroll_to(self.item)
            s = call_bot(text)
            if s:
                self.item = TwoLineListItem(
                    text='Bot',
                    secondary_text=s,)
                self.list_view.add_widget(self.item)
                self.scroll.scroll_to(self.item)
                self.close_button = MDRaisedButton(
                    text='????ng', on_release=self.close_dialog)
                self.dialog = MDDialog(
                    title='K???t qu???', text=s, buttons=[self.close_button])
                self.dialog.open()
        else:
            speak('kh??ng c?? d??? li???u')

    def build(self):
        Window.pos_hint = {'center_x': 0, 'center_y': 0}
        self.state = 0  # initial state
        screen = MDScreen()

        # top toolbar
        self.toolbar = MDToolbar(title="Tr??? l?? ???o")
        self.toolbar.pos_hint = {"top": 1}
        self.toolbar.right_action_items = [
            ["play"]]
        # banner

        # list view
        self.scroll = ScrollView(
            pos_hint={"top": 0.9},
            size_hint=(1, 0.78)
        )
        self.list_view = MDList()
        self.scroll.add_widget(self.list_view)
        self.item2 = OneLineListItem(text='Bot: Ch??o b???n ' + name)
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(text='T??i l?? tr??? l?? VKU')
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(
            text='T??i c?? th??? gi??p b???n r???t nhi???u vi???c ????')
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(text='T??i c?? th??? gi??p b???n d???ch')
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(text='T??i c?? th??? gi??p b???n ch??i nh???c')
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(text='T??i c?? th??? gi??p b???n xem th???i ti???t')
        self.list_view.add_widget(self.item2)
        self.item2 = OneLineListItem(text='V?? r???t nhi???u vi???c n???a h?? h??')
        self.list_view.add_widget(self.item2)
        self.input = MDTextField(
            size_hint=(0.6, 0.4),
            hint_text="B???n mu???n n??i g?? v???i t??i",
            mode="rectangle",
            pos_hint={"center_x": 0.35, "center_y": 0.07},
            font_size=15
        )

        # secondary + primary labels
        # "CONVERT" button
        self.voicemic = MDRaisedButton(
            text="N??i",
            font_size=17,
            size_hint=(0.14, None),
            pos_hint={"center_x": 0.9, "center_y": 0.06},
            on_press=self.voice
        )
        self.send = MDRaisedButton(
            text="G???i",
            font_size=17,
            size_hint=(0.14, None),
            pos_hint={"center_x": 0.75, "center_y": 0.06},
            on_press=self.text
        )
        screen.add_widget(self.scroll)
        screen.add_widget(self.voicemic)
        screen.add_widget(self.send)
        screen.add_widget(self.input)
        screen.add_widget(self.toolbar)

        return screen


if __name__ == '__main__':
    speak('Ch??o b???n ' + name)
    ConverterApp().run()
