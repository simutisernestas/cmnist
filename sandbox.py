import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from mnist import Net
import warnings
warnings.filterwarnings("ignore")


def postprocess(data):
    return data.argmax(1).tolist()


# load model mnist_cnn.pt
model = Net()
model.load_state_dict(torch.load('data/mnist_cnn.pt'))
model.eval()

layers = ["conv1", "conv1_bias", "conv2",
          "conv2_bias", "fc1", "fc1_bias", "fc2", "fc2_bias"]
for i, param in enumerate(model.parameters()):
    print(param.shape, param.dtype)
    param.requires_grad = False
    param.reshape(-1).numpy().tofile(f'data/binm/{layers[i]}.bin')
    # print(param.reshape(-1)[:10])
    # print(param.reshape(-1).shape)
    # print(param[0][0])

image_processing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open('data/0.png')
img = np.array(img)
print(img.shape)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j] == 0:
            print(' ', end='')
        else:
            print('#', end='')
    print()

# inference
img = image_processing(img)
img = img.unsqueeze(0)
output = model(img)
output = postprocess(output)
print(output)
