import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


'''
The Meta learning process for the Meta-Face2Exp, 
The following is a simple pass over one epoch

It does:
 - Teacher forward
 - Student forward
 - Student update
 - Teacher meta update (based on student improvement)
'''


# Dummy Models
def create_model(num_classes):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return model

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

# Create teacher and student models
teacher_model = create_model(num_classes).cuda()
student_model = create_model(num_classes).cuda()

teacher_optimizer = optim.SGD(teacher_model.parameters(), lr=0.01, momentum=0.9)
student_optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)

# Loss function
criterion = nn.CrossEntropyLoss()

# Dummy datasets
labeled_data = TensorDataset(torch.randn(steps_per_epoch * batch_size, 3, 32, 32),
                             torch.randint(0, num_classes, (steps_per_epoch * batch_size,)))
unlabeled_data = TensorDataset(torch.randn(steps_per_epoch * unlabeled_batch_size, 3, 32, 32))

labeled_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_data, batch_size=unlabeled_batch_size, shuffle=True)

for epoch in range(epochs):
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    for step in range(steps_per_epoch):
        images_l, targets = next(labeled_iter)
        (images_uw,) = next(unlabeled_iter)
        images_us = torch.randn_like(images_uw)

        images_l = images_l.cuda()
        targets = targets.cuda()
        images_uw = images_uw.cuda()
        images_us = images_us.cuda()

        teacher_model.train()
        student_model.train()

        batch_size_l = images_l.size(0)

        # Teacher forward
        t_images = torch.cat((images_l, images_uw, images_us), dim=0)
        t_logits = teacher_model(t_images)
        t_logits_l = t_logits[:batch_size_l]
        t_logits_uw, t_logits_us = t_logits[batch_size_l:].chunk(2)

        t_loss_l = criterion(t_logits_l, targets)

        soft_pseudo_label = torch.softmax(t_logits_uw.detach() / temperature, dim=-1)
        max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_probs.ge(threshold).float()

        loss_u = F.cross_entropy(t_logits_us, hard_pseudo_label, reduction='none')
        t_loss_u = (loss_u * mask).mean()

        # Student forward
        s_images = torch.cat((images_l, images_us), dim=0)
        s_logits = student_model(s_images)
        s_logits_l = s_logits[:batch_size_l]
        s_logits_us = s_logits[batch_size_l:]

        s_loss_l_old = F.cross_entropy(s_logits_l.detach(), targets)
        s_loss = criterion(s_logits_us, hard_pseudo_label)

        # Student update
        student_optimizer.zero_grad()
        s_loss.backward()
        student_optimizer.step()

        # Teacher meta update
        with torch.no_grad():
            s_logits_l_new = student_model(images_l)
        s_loss_l_new = F.cross_entropy(s_logits_l_new.detach(), targets)

        dot_product = s_loss_l_old - s_loss_l_new
        t_loss_mpl = dot_product * F.cross_entropy(t_logits_us, hard_pseudo_label)

        t_loss_total = t_loss_l + lambda_u * t_loss_u + t_loss_mpl

        teacher_optimizer.zero_grad()
        t_loss_total.backward()
        teacher_optimizer.step()

    print(f"Epoch {epoch + 1} finished.")

print("Training done.")

