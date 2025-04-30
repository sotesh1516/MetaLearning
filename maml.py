import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')


'''
model = Net()
specify support and query set
for i in range of meta iteration:
    for j in  a pool of tasks:
        for k in samples of a specific task -> adaptation using support set 
            clone a copy of main model parameters
            forward an example 
            compute the loss and gradient
            update current models parameters 
            save the gradient into an array
            return the adapted model
        evaluate on query set and append the loss to meta_losses
    update the main model's gradient based on the average loss -> mean_loss backprop
        
task creation -> i will be manually creating different tasks/ mini batches of datasets depending on distributions,.. since I am just starting to do meta learning
              -> create a large pool of tasks, maybe like a hundred so each meta iteration gets a new task
              
most imbalance vs performance

- outline what the training process would be
        - we are gonna train this model
        - we are gonna try these loss function in the inner loop
        - we are gonna try these loss function in the outer loop
- Which can be a baseline, possble papers?
        - paper 1
        - paper 2
        - paper 3
- What do we want as a part of our inner loop? What can be a good objective function to address the bias?
        - For Meta-Face2Exp, it was using external dataset
        - Maybe do something on the minority class where I do sometype of self-supervised objective, classification on just that minority class and learn weights to
          update the overall objective, 
                - possible objectives
                    - assigning weight to each class -> class weights = [1.0, 1.0, 3.0, 1.0] and then apply cross entropy
                    - focal loss -> modified cross entropy, focus on learning hard, which is super useful for imbalanced datasets.


cvpr
icc
acl

project with PHD student
reach out to an advisor and undergraduate research
'''

