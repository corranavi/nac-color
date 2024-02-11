#Questo è un file di prova per verificare che effettivamente le trasformazioni siano applicate
#in modo uguale a tutte le slice di una sequenza

import torch
from torchvision.transforms import v2 as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from transformations import TRANSFORMATIONS

image = torch.randn((3,224,224))
cloned = image.clone()
print((image.unsqueeze(0)).shape) #(1,3,15,15)

# image = Image.open("C:\\Users\\c.navilli\\Pictures\\bimbo_piange.png")
# cloned = Image.open("C:\\Users\\c.navilli\\Pictures\\bimbo_piange.png")
# prima_trasf = T.Compose([T.ToTensor()])
# image = prima_trasf(image)[:3,:,:]
# cloned = prima_trasf(cloned)[:3,:,:]

print(f"Image 1 shape: {image.shape}")
print(f"Image 2 shape: {cloned.shape}")

#concatena
unsqueeze = True
if unsqueeze:
    both_images = torch.cat((image.unsqueeze(0), cloned.unsqueeze(0)), 0) 
else:
    both_images = torch.cat((image,cloned),0) #errato, mi va a sommare le immagini sui canali

print(both_images.shape)

#transformed_images = T.RandomRotation(0)(both_images)

transforms = T.Compose([
    T.RandomResizedCrop(size=(224, 224), antialias=True),
    T.RandomHorizontalFlip(p=0.5),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transformed_images = transforms(both_images)


# trasformazioni=TRANSFORMATIONS["train"]
# print(trasformazioni)
# transformed_images = trasformazioni(image=both_images)

# get transformed images 
image_tx = transformed_images[0]
cloned_tx = transformed_images[1]

#check by print
print(image_tx.shape)
print(cloned_tx.shape)

image = np.clip(image, 0, 1)
cloned_tx = np.clip(cloned_tx, 0, 1)

plt.imshow(image.view(image.shape[1], image.shape[2], image.shape[0]))
plt.show()
plt.imshow(cloned_tx.view(cloned_tx.shape[1],cloned_tx.shape[2],cloned_tx.shape[0]))
plt.show()

print(torch.all(image_tx == cloned_tx).item())

#Ok tutto funziona ---> la trasformazione viene applicata in modo uguale a tutte le slice della
#sottosequenza