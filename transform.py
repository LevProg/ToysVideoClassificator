from torchvision import transforms
import torch
import PIL

valid_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def Transform(image):
    image=valid_transform(image)
    return image