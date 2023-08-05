import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from mnist import Net
import warnings
warnings.filterwarnings("ignore")

# load model mnist_cnn.pt
model = Net()
model.load_state_dict(torch.load('data/mnist_cnn.pt'))
model.eval()

image_processing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img = Image.open('data/0.png')
img = np.array(img)

img = image_processing(img)
img = img.unsqueeze(0)
out = model.conv1(img)
pytorch_out = out.reshape(-1).detach().numpy()
C_conv1_out = np.fromfile('log/conv1_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_conv1_out.shape
assert np.allclose(pytorch_out, C_conv1_out, atol=1e-6)

out = F.relu(out)
pytorch_out = out.reshape(-1).detach().numpy()
C_relu_out = np.fromfile('log/conv1_relu_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_relu_out.shape
assert np.allclose(pytorch_out, C_relu_out, atol=1e-6)

out = model.conv2(out)
pytorch_out = out.reshape(-1).detach().numpy()
C_conv2_out = np.fromfile('log/conv2_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_conv2_out.shape
assert np.allclose(pytorch_out, C_conv2_out,
                   atol=1e-6), np.linalg.norm(pytorch_out - C_conv2_out)

out = F.max_pool2d(out, 2)
pytorch_out = out.reshape(-1).detach().numpy()
C_max_pool2d_out = np.fromfile('log/max_pool2d_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_max_pool2d_out.shape
assert np.allclose(pytorch_out, C_max_pool2d_out, atol=1e-6)

# fc1
out = model.fc1(torch.flatten(out, 1))
pytorch_out = out.squeeze(0).detach().numpy()
C_fc1_out = np.fromfile('log/fc1_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_fc1_out.shape, (pytorch_out.shape, C_fc1_out.shape)
assert np.allclose(pytorch_out, C_fc1_out, atol=1e-6)

# relu
out = F.relu(out)
pytorch_out = out.squeeze(0).detach().numpy()
C_relu_out = np.fromfile('log/fc1_relu_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_relu_out.shape
assert np.allclose(pytorch_out, C_relu_out, atol=1e-6)

# fc2
out = model.fc2(out)
pytorch_out = out.squeeze(0).detach().numpy()
C_fc2_out = np.fromfile('log/fc2_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_fc2_out.shape
assert np.allclose(pytorch_out, C_fc2_out, atol=1e-6)


