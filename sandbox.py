import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
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
out = model.conv1(img)
out = F.relu(out)
for i in range(32):
    print(out[:, i, :3, :10])
# int src_base = m * conv1_out_height * conv1_out_width + j * conv1_out_width + k;

# 3 + 3*26 + 26
print('conv1 out:', out.shape)
out = model.conv2(out)
print('conv2 out:', out.shape)
for p in out.reshape(-1)[:10].tolist():
    print(p)
# print(out.shape)
print("conv2 weights")
print(model.conv2.weight.shape)
# for w in model.conv2.weight.reshape(-1)[:10].tolist():
#     print(w)
for b in model.conv2.bias.reshape(-1)[:10].tolist():
    print(b)
print(model.conv2.bias.shape)
print("conv1 weights")
print(model.conv1.weight.shape)
print(model.conv1.bias.shape)

# output = model(img)
# output = postprocess(output)
# print(output)

# if __name__ == '__main__':
#     x = torch.ones(1,1,28,28)
#     weight = torch.rand(1, 1, 3, 3)
#     bias = torch.rand(1)
#     conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1)
#     conv1.weight.data = weight
#     conv1.bias.data = bias
#     y1 = conv1(x)
#     print(y1.shape)

#     # convolve in raw python
#     x = x.squeeze(0).squeeze(0)
#     y2 = torch.zeros(26, 26)
#     for i in range(26):
#         for j in range(26):
#             print(x[i:i+3, j:j+3])
#             print(weight)
#             print((x[i:i+3, j:j+3] * weight))
#             exit()
#             y2[i][j] = (x[i:i+3, j:j+3] * weight).sum() + bias
#     y2.unsqueeze(0).unsqueeze(0)

#     # check if close
#     print(torch.allclose(y1, y2))
