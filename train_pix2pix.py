import torch
from torch.utils.data import DataLoader
from model_pix2pix import Generator, Discriminator
from dataset_loader import SketchDataset

dataset = SketchDataset("dataset")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

G = Generator()
D = Discriminator()

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

criterion = torch.nn.BCEWithLogitsLoss()

for epoch in range(10):

    for sketch, photo in loader:

        # Train Generator
        fake_photo = G(sketch)

        pred_fake = D(sketch, fake_photo)

        loss_G = criterion(pred_fake, torch.ones_like(pred_fake))

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # Train Discriminator
        pred_real = D(sketch, photo)
        loss_real = criterion(pred_real, torch.ones_like(pred_real))

        pred_fake = D(sketch, fake_photo.detach())
        loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_real + loss_fake) / 2

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    print("Epoch:", epoch)