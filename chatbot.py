import json
import os
import random
import time
import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from webdriver_manager.chrome import ChromeDriverManager
from gtts import gTTS
import playsound
import speech_recognition
import pyttsx3
from datetime import date, datetime
import wikipedia
# import ZaloTTSMouth
from zalo_tts import ZaloTTS
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()

robot_ear = speech_recognition.Recognizer()  # tao doi tuong nghe
robot_mouth = pyttsx3.init()  # tao doi tuong noi
robot_brain = ""  # tao doi tuong hieu

stemmer = PorterStemmer()
you_text = ""


# Xử lý phần nói
def speak(text):
    print("Robot:" + text)
    tts = gTTS(text=text, lang=language, slow=False)
    date_string = datetime.now().strftime("%d%m%Y%H%M%S")
    n = "text"+date_string+".mp3"
    tts.save(n)
    playsound.playsound(n, True)
    os.remove(n)


def stop():
   speak("Tạm biệt")


def time_now():
    return datetime.now()


def mic():
    you_text = " "
    with speech_recognition.Microphone() as mic:
        # print("-- Robot on --")
        # time = time_now()
        # if time.hour > 0 and time.hour < 12:
        #     ZaloTTSMouth.ZaloMouth("Chào buổi sáng")
        # elif time.hour >= 12 and time.hour < 18:
        #     ZaloTTSMouth.ZaloMouth("Chào Buổi chiều ")
        # elif time.hour >= 18 and time.hour < 24:
        #     ZaloTTSMouth.ZaloMouth("Chào buổi tối")
        #
        # ZaloTTSMouth.ZaloMouth("Tôi là Mai")
        audio = robot_ear.listen(mic, phrase_time_limit=3)  # phrase_time_limit: thời gian chờ sau khi dừng nói

        try:
            you_text = robot_ear.recognize_google(audio, language='vi-VN')
            print(you_text)
            return you_text
        except:
            return "AHDKDNRJEYEIAJDBS"



# def get_text():
#     for i in range(3):
#         text = you_text
#         if text:
#             return speak(text)
#         elif i < 2:
#             speak("Tôi không nghe rõ. Bạn nói lại được không!")
#     time.sleep(3)
#     stop()
#     return 0


# ____________________________________________


# Chức năng giao tiếp chào hỏi
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(s, words):
    # stem each word
    sentence_words = [stem(word) for word in s]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out


# Version 2.0 - AI ChatBot
file = open('intents.json', encoding="utf8")
with file as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print(len(xy), "patterns")
# print(len(tags), "tags:", tags)
# print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # AMD not NVIDIA
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. Save to {FILE}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', encoding="utf8") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

#
# def chat():
#     print("-- Robot on --")
#     time = time_now()
#     if time.hour > 0 and time.hour < 12:
#         ZaloTTSMouth.ZaloMouth("Chào buổi sáng")
#     elif time.hour >= 12 and time.hour < 18:
#         ZaloTTSMouth.ZaloMouth("Chào Buổi chiều ")
#     elif time.hour >= 18 and time.hour < 24:
#         ZaloTTSMouth.ZaloMouth("Chào buổi tối")
#     ZaloTTSMouth.ZaloMouth("Tôi là Mai")
#     ZaloTTSMouth.ZaloMouth("Tôi có thể giúp gì cho bạn?")
#     print("Let's talk! (say 'Kết thúc' or 'stop' to exit)")
#
#     while True:
#         mic()
#         # sentence = "do you use credit cards?"
#         sentence = you_text
#         print(sentence)
#         if "kết thúc" in sentence or 'stop' in sentence:
#             speak("Kết thúc trò chuyện")
#             break
#
#         sentence = tokenize(sentence)
#         X = bag_of_words(sentence, all_words)
#         X = X.reshape(1, X.shape[0])
#         X = torch.from_numpy(X).to(device)
#
#         output = model(X)
#         _, predicted = torch.max(output, dim=1)
#
#         tag = tags[predicted.item()]
#
#         probs = torch.softmax(output, dim=1)
#         prob = probs[0][predicted.item()]
#         print(prob.item())
#
#         for intent in intents['intents']:
#             if tag == intent["tag"]:
#                 ZaloTTSMouth.ZaloMouth(random.choice(intent['responses']))


def chat():
    print("-- Robot on --")
    time = time_now()
    if time.hour > 0 and time.hour < 12:
        speak("Chào buổi sáng")
    elif time.hour >= 12 and time.hour < 18:
        speak("Chào Buổi chiều ")
    elif time.hour >= 18 and time.hour < 24:
        speak("Chào buổi tối")
    speak("Tôi là Mai")
    speak("Tôi có thể giúp gì cho bạn?")
    print("Let's talk! (say 'Kết thúc' or 'stop' to exit)")

    while True:

        text = mic()
        # sentence = "do you use credit cards?"
        sentence = text
        print(text)
        print("XUẤT SEN")
        print(sentence)
        if "kết thúc" in sentence or 'stop' in sentence:
            speak("Kết thúc trò chuyện")
            break
        elif "AHDKDNRJEYEIAJDBS" in sentence:
            speak("Tôi không nghe được bạn nói gì? Vui lòng nói lại")
            continue
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        print("NOTE tại prob")
        print(prob.item())

        for x in intents['intents']:
            if tag == x["tag"]:
                speak(random.choice(x['responses']))

            if intent in ["Bây giờ là mấy giờ", "thời gian hiên tại"]:
                # Thay thế date vào câu trả lời của intent
                for a in intent["responses"]:
                    response_with_date = a.replace("{{date}}", date)
                    print(response_with_date)



# /Phan lang nghe

# Demo xử lý phân doc hiểu cua chatbot --- Trí tuệ nhân tạo
# while True:
#     if you_text == "":
#         robot_brain = "Tôi không hiểu bạn nói gì! Vui lòng nói lại"
#     elif "goodbye" in you_text or "tạm biệt" in you_text or "kết thúc" in you_text or "stop" in you_text:
#         stop()
#         break
#     elif "Xin chào" in you_text:
#         robot_brain = "Tôi vẫn đang lắng nghe"
#     elif "ngày hôm nay" in you_text:
#         today = date.today()
#         robot_brain = today.strftime("%B %d. %Y")
#     elif "mấy giờ" in you_text:
#         now = datetime.now()
#         # Chuyển đổi sang tiếng Việt
#         hours = int(now.strftime("%H"))
#         minutes = int(now.strftime("%M"))
#         seconds = int(now.strftime("%S"))
#
#         robot_brain = f"{hours} giờ {minutes} phút {seconds} giây"
#         # robot_brain = now.strftime(f"%H hours %M minutes %S seconds")
#




chat()
