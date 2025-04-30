import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader
from torchvision import transforms

# Hyperparameters
num_classes = 7
batch_size = 64
unlabeled_batch_size = batch_size * 2
threshold = 0.95
temperature = 1.0
lambda_u = 1.0
epochs = 1
steps_per_epoch = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_classes):
    model = None
    
    return model

# Create teacher and student models
base_model = create_model(num_classes).cuda()
adaptation_model = create_model(num_classes).cuda()

base_optimizer = optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9)
adaptation_optimizer = optim.SGD(adaptation_model.parameters(), lr=0.01, momentum=0.9)

# Loss function
criterion = nn.CrossEntropyLoss()

#Dataloaders
fer_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

fer_dataset = ImageFolder(root='dataset/fer_data', transform=fer_transform)

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = transform
        self.loader = default_loader
    
    def __getitem__(self, idx):
        img = self.loader(self.paths[idx])
        return self.transform(img)

    def __len__(self):
        return len(self.paths)

fr_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

fr_dataset = UnlabeledDataset('dataset/fr_data/train/all_images', transform=fr_transform)

batch_size = 64
unlabeled_batch_size = batch_size * 2

fer_loader = DataLoader(fer_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
fr_loader = DataLoader(fr_dataset, batch_size=unlabeled_batch_size, shuffle=True, num_workers=4)


for epoch in range(epochs):
    labeled_iter = iter(fer_loader)
    unlabeled_iter = iter(fr_loader)
    
    for step in range(steps_per_epoch):
        #fetch one mini batch of FER data
        #tuple of image and labels
        images_lab, targets = next(labeled_iter)
        #fetch one mini batch of FR data
        (images_unlab,) = next(unlabeled_iter)
        
        images_lab = images_lab.to(device)
        images_unlab = images_unlab.to(device)
        
        
        base_model.train()
        adaptation_model.train()
        
        batch_size_lab = images_lab.size(0)
        
        #base network forward
        base_pass_images = torch.cat((images_lab, images_unlab), dim=0)
        #base logit dim (batch_size_lab + batch_size_lab * 2) x num_classes
        base_logits = base_model(base_pass_images)
        base_logit_lab = base_logits[:batch_size_lab]
        base_logit_unlab = base_logits[batch_size_lab:]
        
        base_super_loss = criterion(base_logit_lab, targets)
        
        #a probability distribution for each image
        soft_pseudo_label = torch.softmax(base_logit_unlab.detach() / temperature, dim=-1)
        
        # hard_pseudo_label picks the class index with high prob
            # need it to train the adaptation network
        # max_probs saves the value of the highest probabilty
            # not all pseudo labels are reliable, if the model is unsure, given a threshold
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        #return a boolean tensor for each sample
        mask = max_probs.ge(threshold).float()
    

