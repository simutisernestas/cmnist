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

# relu
out = F.relu(out)

# conv2
out = model.conv2(out)
pytorch_out = out.reshape(-1).detach().numpy()
C_conv2_out = np.fromfile('log/conv2_out.bin', dtype=np.float32)
assert pytorch_out.shape == C_conv2_out.shape
assert np.allclose(pytorch_out, C_conv2_out, atol=1e-3)