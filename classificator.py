import cv2
import pickle
import torch

classDict= {0:'Муравьед',
            1:'Коала',
            2:'Плащевидная ящерица',
            3:'Крокодил',
            4:'Дельфин',
            5:'Лягушки',
            6:'Осьминог',
            7:'Попугай',
            8:'Черепаха',
            9:'Вомбат',
            10:'Кенгуру',
            11:'Обезьяна',
            12:'Пустой фон'}


model=torch.load('model.pt')
model.eval()
if torch.cuda.is_available():
    model=model.cuda()
    print('Есть Cuda')
else:
    model=model.cpu()
    print('Нет Cuda')

def classificate(image):
    if torch.cuda.is_available():
        image = image.cuda()
    else:
        image = image.cpu()
    pred = model(image[None, ...])
    classIndex=pred.argmax().item()
    return classDict[classIndex]