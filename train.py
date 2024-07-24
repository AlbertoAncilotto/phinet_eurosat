import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import numpy as np
from torch.utils.data import DataLoader, Dataset
from glob import glob
from torchsummary import summary
import os
import random
import itertools
from tqdm import tqdm
import argparse
import tifffile

parser = argparse.ArgumentParser(description='EuroSat PhiNet training.')
parser.add_argument('--zero_channels', type=int, nargs='+', default=[], help='List of zero channels.')
parser.add_argument('--net_name', type=str, default='model', help='Model name.')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
parser.add_argument('--train_fraction', type=float, default=1.0, help='Training data percentage.')
parser.add_argument('--train_unbalanced', type=bool, default=False, help='Use unbalanced data for training (real life distribution).')
parser.add_argument('--input_dropout', type=bool, default=False, help='Randomly drop one channel at each step for augmentation.')
parser.add_argument('--model', type=str, default='phinet', help='Options: ["phinet", "resnet"]')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_CHANNELS = 13
ZERO_CHANNELS = args.zero_channels
NUM_EPOCHS = args.epochs
NET_NAME = args.net_name
TRAIN_FRACTION = args.train_fraction

root = "dataset_allbands/ds/images/remote_sensing/otherDatasets/sentinel_2/tif"
images_paths = [glob(f'{root}/{folder}/*.tif') for folder in os.listdir(f"{root}")] 
images_paths = list(itertools.chain.from_iterable(images_paths))
random.Random(42).shuffle(images_paths) #seeded shuffle

class EuroSatTifDataset(Dataset):
    
    def __init__(self, root, images_paths, train=True, transform_status=True, zero_channels=[], class_probabilities=None, fraction=1.0):
        self.root = root
        self.train = train
        self.transform_status = transform_status
        self.zero_channels = zero_channels
        self.transform = transforms.Normalize(mean=[0.5]*INPUT_CHANNELS, std=[0.225]*INPUT_CHANNELS)
        self.class_probabilities = class_probabilities
        self.fraction = fraction
        
        split_index_start = 0 if self.train else int(0.85 * len(images_paths))
        split_index_stop = int(0.85 * len(images_paths)) if self.train else len(images_paths)
        self.images_paths = images_paths[split_index_start:split_index_stop]

        if self.fraction<1.0:
            self.images_paths = self.images_paths[:int(len(self.images_paths) * self.fraction)]

        self.classes_names = {class_name:label for label, class_name in enumerate(os.listdir(f"{root}"))}
        self.labels = [self.classes_names[os.path.basename(os.path.dirname(path))] for path in self.images_paths]

        if self.class_probabilities is not None:
            self.cumulative_probabilities = np.cumsum(self.class_probabilities)
        else:
            self.cumulative_probabilities = None
    
    def __len__(self):
        return len(self.images_paths)
    
    def __getitem__(self, index):
        if self.cumulative_probabilities is not None:
            rand_value = np.random.rand()
            class_index = np.searchsorted(self.cumulative_probabilities, rand_value)
            class_indices = [i for i, label in enumerate(self.labels) if label == class_index]
            index = random.choice(class_indices)
        
        image_path = self.images_paths[index]
        image = tifffile.imread(image_path)
        image = torch.tensor((image/7500).astype(np.float32))
        image = image.permute(2, 0, 1)
        
        if self.transform_status: 
            image = self.transform(image)
            for ch in self.zero_channels:
                image[ch, :, :] = 0
        
        if self.train:
            if args.input_dropout:
                random_channel = np.random.randint(INPUT_CHANNELS)
                image[random_channel, :, :] = 0
            rotation_angle = random.choice([0, 90, 180, 270])
            image = torch.rot90(image, rotation_angle // 90, (1, 2))
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=(2,))
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=(1,))
        
        label = self.labels[index]
        return image.float(), torch.tensor([label]).float()

class_probabilities = np.array([0.11, 0.31, 0.14, 0.01, 0.01, 0.27, 0.01, 0.03, 0.01, 0.10]) if args.train_unbalanced else None
train_dataset = EuroSatTifDataset(root, images_paths, train=True, transform_status=True, zero_channels=ZERO_CHANNELS, class_probabilities=class_probabilities)
test_dataset = EuroSatTifDataset(root, images_paths, train=False, transform_status=True, zero_channels=ZERO_CHANNELS, class_probabilities=class_probabilities)
train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
test_dataset_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)

def build_model():
    if args.model == 'phinet':
        from micromind.networks import PhiNet
        model = PhiNet( (INPUT_CHANNELS, 64, 64),
                        alpha=0.5,
                        beta=0.75,
                        t_zero=5,
                        num_layers=8,
                        h_swish=False,
                        squeeze_excite=True,
                        include_top=True,
                        num_classes=1000,
                        divisor=8,
                        compatibility=False)
        model = nn.Sequential(model,
                            nn.ReLU(),
                            nn.Linear(1000, 10, bias=True,))
    
    elif args.model == 'resnet':
        model = models.resnet50(pretrained=False)
        model.conv1 = nn.Conv2d(13, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    return model.to(device) , loss_fn, optimizer

model, loss_function, optimizer = build_model()
summary(model,(INPUT_CHANNELS, 64, 64))

def train_batch(model, loss_function, optimizer, image, label):
    model.train()
    optimizer.zero_grad()
    prediction = model(image.to(device))
    loss = loss_function(prediction, label.long().squeeze().to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def accuracy(model, loss_function, image, label):
    model.eval()
    prediction = model(image.to(device))
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == label.long().squeeze().to(device)
    return is_correct.cpu().numpy().tolist(), argmaxes

@torch.no_grad()
def validation_loss(model, loss_function, image, label):
    model.eval()
    prediction = model(image.to(device))
    loss = loss_function(prediction, label.long().squeeze().to(device))
    return loss.item()

best_accuracy = 0.0
best_checkpoint_path = f'checkpoints/{NET_NAME}.pt'

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch+1}")
    train_epoch_losses = []
    train_epoch_accuracies = []
    for image, label in tqdm(iter(train_dataset_loader)):
        loss = train_batch(model, loss_function, optimizer, image, label)
        train_epoch_losses.append(loss)
    train_epoch_loss = np.mean(train_epoch_losses)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    
    test_epoch_losses = []
    test_epoch_accuracies = []
    for image, label in tqdm(iter(test_dataset_loader)):
        loss = validation_loss(model, loss_function, image, label)
        is_correct, predicted_classes = accuracy(model, loss_function, image, label)
        test_epoch_losses.append(loss)
        test_epoch_accuracies.extend(is_correct)
    test_epoch_loss = np.mean(test_epoch_losses)
    print(f"Test Loss: {test_epoch_loss:.4f}")
    test_epoch_accuracy = np.mean(test_epoch_accuracies)
    print(f"Test Accuracy: {test_epoch_accuracy*100:.2f}%")

    if test_epoch_accuracy > best_accuracy:
        best_accuracy = test_epoch_accuracy
        torch.save(model.state_dict(), best_checkpoint_path)

with open(f'checkpoints/{NET_NAME}_{best_accuracy}.log', 'w') as f:
    f.write('best acc: '+str(best_accuracy))