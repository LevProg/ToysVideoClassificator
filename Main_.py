import cv2
import torch
from torchvision import transforms
from tkinter import *
from PIL import ImageTk
import threading
import PIL



label = None

valid_transform = transforms.Compose([#Трансформы которые мы будем применять к входящим изображениям
    transforms.ToPILImage(),
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def Transform(image):
    image = valid_transform(image)#Производим трансформ
    return image


classDict = {0: 'Муравьед',#Словарь из номера класса:имени
             1: 'Коала',
             2: 'Плащевидная ящерица',
             3: 'Крокодил',
             4: 'Дельфин',
             5: 'Лягушки',
             6: 'Осьминог',
             7: 'Попугай',
             8: 'Черепаха',
             9: 'Вомбат',
             10: 'Кенгуру',
             11: 'Обезьяна',
             12: 'Пустой фон'}

def classificate(image):# тут мы получаем изображением, трансформим его
    image=Transform(image)#, кормим модельке, в ответ получаем тензор в котором выбираем самый вероятный класс
    image = image.cpu()
    pred = model(image[None, ...])
    classIndex = pred.argmax().item()
    return classDict[classIndex]#Возвращаем название класса

def window():#Здесь мы создаём окно, настраиваем его
    root = Tk()
    
    root.title("Classificator")

    global label, lmain
    label = Label(root, text='', fg='black')#label предназначенный для вывода предсказания
    lmain= Label(root)#label предназначенный для вывода изображения с вебкамеры
    RButton=Button(root, text="Классифицировать в реальном времени",#Button  для включения распознавания изображения
                   background='Grey', cursor="target" ,
                   padx=10, command=threadPhoto.start)

    lmain.pack( ipadx=10,
                ipady=10,
                fill='both')
    RButton.pack( ipadx=10,
                ipady=10,   #Упаковка компонентов окна
                fill='both')
    label.pack( ipadx=10,
                ipady=10,
                fill='both')
    

    x = root.winfo_screenwidth()  # размер  по горизонтали
    y = root.winfo_screenheight()  # размер по вертикали
    root.geometry('{}x{}'.format(int(x * 0.8), int(y * 0.8)))#размер окошка
    show_frame()
    root.mainloop()

def RealTimeClassificate():#Получаем изображение, кидаем в классификатор и выводим ответ
    while True:
        x=classificate(photo_frame())
        print(x)
        mytext = x
        label.config(text=mytext)


def show_frame():#Сбор изображения с камеры и показ на lable
    ret, frame = webcam.read()
    if ret:
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_frame)
def photo_frame():
    ret, frame = webcam.read()
    frame = cv2.flip(frame, 1)
    return frame


key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)

model = torch.load('model1.pt')#загрузка модели
model.eval()
model = model.cpu()


thread = threading.Thread(target=window)#создание потока с окошком
thread.start()

threadPhoto = threading.Thread(target=RealTimeClassificate)#создание потока для классификации 