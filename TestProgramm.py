import cv2
import torch
from torchvision import transforms

torch.clear_autocast_cache()

#Для проверки работоспособности основной программы 

def classificate(image):
    image = image.cpu()
    pred = model(image[None, ...])
    classIndex = pred.argmax().item()
    return classDict[classIndex]

def Transform(image):
    image = valid_transform(image)
    return image

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

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

model = torch.load('model.pt')
model.eval()
i_1 =cv2.imread('3.jpg')
model = model.cpu()
print(classificate(Transform(i_1)))

