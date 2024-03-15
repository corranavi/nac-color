import torch
from models import MultiParametricMRIModel

def main():
    x = torch.randn(8, 3, 224, 224) 
    print("Print of dimensions of a sample (8 slices, one for each modality AND pre-post)")
    print(f"Full shape: {x.shape}")
    print(f"Shape of a single slice: {x[0].shape}")
    print(f"Shape of a single slice (proper input for ResNet): {x[0].unsqueeze(0).shape}\n")

    model = MultiParametricMRIModel()
    model.eval()
    probs = model(x)

    for k,v in probs.items():
        print(f"{k}: \nProb class 0: {v[0][0]}\nProb class 1: {v[0][1]}")
        print(f"\tPredicted: {v.argmax().item()}\n")

if __name__=="__main__":
    main()