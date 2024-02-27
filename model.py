import os
from torchvision.models import resnet50, ResNet50_Weights

os.environ["TORCH_HOME"] #Instancing a pre-trained model will download its weights to a cache directory. This directory can be set using the TORCH_HOME environment variable
#resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


model = resnet18(pretrained=False)
model.fc = Identity()
x = torch.randn(1, 3, 224, 224)
output = model(x)
print(output.shape)