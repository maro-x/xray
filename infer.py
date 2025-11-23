import torch
from torchvision import transforms, models
from torch import nn
import pickle
from PIL import Image



with open(r"C:/Users/DELL/OneDrive/Documents/GitHub/xray/data/lbl_encoder.pkl", "rb") as f:
    lbl_encoder = pickle.load(f)



inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])



def load_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(
        r"C:/Users/DELL/OneDrive/Documents/GitHub/xray/models/model_resnet18.pth",
        map_location="cpu"
    ))
    model.eval()
    return model



def load_resnet34():
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(
        r"C:/Users/DELL/OneDrive/Documents/GitHub/xray/models/model_resnet34.pth",
        map_location="cpu"
    ))
    model.eval()
    return model



def load_mobilenet():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(1280, 2)
    model.load_state_dict(torch.load(
        r"C:/Users/DELL/OneDrive/Documents/GitHub/xray/models/model_mobilenet_v2.pth",
        map_location="cpu"
    ))
    model.eval()
    return model


model_resnet18 = load_resnet18()
model_resnet34 = load_resnet34()
model_mobilenet = load_mobilenet()



def infere(img, model_name="resnet18"):

    img_gray = img.convert("L")
    img_trans = inference_transform(img_gray).unsqueeze(0)

    # Select model
    if model_name == "resnet18":
        model = model_resnet18
    elif model_name == "resnet34":
        model = model_resnet34
    elif model_name == "mobilenet":
        model = model_mobilenet
    else:
        raise Exception("Unknown model!")

    # forward
    with torch.no_grad():
        preds = model(img_trans)
        probs = torch.softmax(preds, dim=1)

        clas_index = torch.argmax(probs).item()
        final_pred = lbl_encoder.inverse_transform([clas_index])[0]

    print(f"Model: {model_name} | Probabilities:", probs.numpy())
    return final_pred


# Example
img_path = r'F:\New folder (2)\chest_xray\chest_xray\train\NORMAL\IM-0129-0001 - Copy - Copy.jpeg'
img = Image.open(img_path)
print(infere(img,'mobilenet'))
