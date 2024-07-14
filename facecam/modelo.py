import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

# Clase SmileDetector para definir el modelo
class SmileDetector(nn.Module):
    def __init__(self):
        super(SmileDetector, self).__init__()
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)  # Entrada RGBA
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv_final = nn.Conv2d(64, 1, kernel_size=1)  # Salida con una sola canal (máscara binaria)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.conv_final(x))  # Usar sigmoide para la máscara de salida (0 a 1)

        return x

# Definición del dataset
class SmileDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        with open(annotations_file) as f:
            self.img_labels = json.load(f)['images']
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_info = self.img_labels[idx]

        print(f"Accediendo a la imagen en el índice: {idx}")
        print(f"Información de la imagen: {img_info}")

        img_path = os.path.join(self.img_dir, img_info['filename'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No se encontró el archivo: {img_path}")

        image = Image.open(img_path).convert("RGBA")

        mask = Image.new('L', (image.size[0], image.size[1]), 0)
        mask_draw = ImageDraw.Draw(mask)
        for region in img_info['regions']:
            points = list(zip(region['shape_attributes']['all_points_x'], region['shape_attributes']['all_points_y']))
            mask_draw.polygon(points, outline=1, fill=1)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Función para entrenar el modelo
def train_smile_detector(dataset_path, annotations_file, model_save_path, epochs=10, batch_size=8, learning_rate=0.001):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    dataset = SmileDataset(annotations_file, dataset_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmileDetector()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, masks in dataloader:
            if torch.cuda.is_available():
                images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}")

    torch.save(model.state_dict(), model_save_path)
    print(f"Modelo guardado en {model_save_path}")

# Entrenar el modelo
if __name__ == "__main__":
    dataset_path = "C:/django/proyecto/dataset/images"  # Reemplaza con la ruta a tus imágenes
    annotations_file = "C:/django/proyecto/dataset/Dataset_json.json"  # Reemplaza con la ruta a tu archivo JSON de anotaciones
    model_save_path = "C:/django/proyecto/smile_detector_model.pth"  # Ruta donde se guardará el modelo entrenado

    train_smile_detector(dataset_path, annotations_file, model_save_path)
