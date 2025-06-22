import streamlit as st
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
latent_dim = 20
img_size = 28

# Model definition (same as training)
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(img_size*img_size + num_classes, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 400)
        self.fc4 = nn.Linear(400, img_size*img_size)

    def encode(self, x, c):
        inputs = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(inputs))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c):
        inputs = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(inputs))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

@st.cache(allow_output_mutation=True)
def load_model():
    model = CVAE().to(device)
    model.load_state_dict(torch.load('cvae_mnist.pth', map_location=device))
    model.eval()
    return model

model = load_model()

st.title("Handwritten Digit Generator (0-9)")

digit = st.slider("Select digit to generate:", 0, 9, 0)

num_images = 5

c = torch.zeros(num_images, num_classes).to(device)
c[:, digit] = 1

z = torch.randn(num_images, latent_dim).to(device)
with torch.no_grad():
    generated = model.decode(z, c).cpu().numpy()

st.write(f"Generated 5 images of digit {digit}:")

cols = st.columns(num_images)

for i, col in enumerate(cols):
    img_array = (generated[i].reshape(img_size, img_size) * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')
    col.image(img, width=100)
