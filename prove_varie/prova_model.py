from torchvision import models
import torch
import torch.nn as nn
import numpy as np

def compute_entropy(values):
    entropy=0
    numpy_array = values.detach().numpy()
    for i in numpy_array:
        entropy += (-1)*(i)*np.log2(i)
    return entropy

def compute_accuracy(preds, labels):
    correct=0
    for i in range(len(preds)):
        if preds[i]==labels[0][i].item():
            correct+=1
    print(correct)
    return float(correct)/len(preds)

classi_dict = {
    0:"Banana",
    1:"Carota",
    2:"Fragola",
    3:"Tristezza infinita"
}

model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)

#print(model)

model.fc = nn.Linear(2048, 4)
x = torch.randn(10, 3, 224, 224) 
y = torch.randint(high=4, size=(1,10))
#print(len(y))
output = model(x)
print(output.shape)

preds=[]
for idx in range(x.shape[0]):
    print(f"Processing image number {idx+1}:")
    img = x[idx].unsqueeze(0)                        #ResNet vuole in input un tensore 4D, quindi devo aggiungere ad ogni immagine un canale a una D
    print(img.shape)
    prediction = model(img).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    preds.append(int(class_id))
    probability = prediction[class_id].item()
    print(f"Le probabilità sono: {prediction}.\nPredicted: {classi_dict[class_id]} con probabilità {probability}")
    indice = y[0][idx].item()
    print(f"Ground truth: {classi_dict[indice]}\n")
    #print(f"La predizione ha il seguente livello di entropia: {compute_entropy(prediction)}\n")

print(preds)
print(list(y[0].detach().numpy()))
print("")
acc = compute_accuracy(preds, y)
print(f"Final accuracy: {acc}")