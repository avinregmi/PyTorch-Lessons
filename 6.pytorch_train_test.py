import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784,256)
    self.fc2 = nn.Linear(256,128)
    self.fc3 = nn.Linear(128,64)
    self.fc4 = nn.Linear(64,10)

    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    #Flatten the input tensor
    x = x.view(x.shape[0],784)

    #Add dropout
    x = self.dropout(F.relu(self.fc1(x))) 
    x = self.dropout(F.relu(self.fc2(x)))
    x = self.dropout(F.relu(self.fc3(x)))

    #Output, so not dropout
    x = F.log_softmax(self.fc4(x), dim=1)

    return x
    

model = Classifier()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.NLLLoss()

epochs = 30
train_losses, test_losses = [], []

for e in range(epochs):
  running_loss = 0
  for images, labels in trainloader:

    #clear the gradient in optimizer
    optimizer.zero_grad()

    #feedforward in the model
    log_ps = model(images)

    #calculate the loss
    loss = criterion(log_ps, labels)

    #compute the loss gradient
    loss.backward()

    #Optimizer to update the weights based on the gradients
    optimizer.step()

    # Add the loss to the training set's rnning loss
    running_loss += loss.item()
  else:
    # Evaluating the model

    accuracy = 0
    test_loss = 0

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
      
      #Turn off dropout in our model for evaluation by setting p=0 and seeting model into eval mode
      model.eval()

      for images, labels in testloader:

        #feedforward our model for validation
        log_ps = model(images)

        # calculate the loss
        test_loss += criterion(log_ps,labels)

        # Since our model outputs a LogSoftmax, find the real 
        # percentages by reversing the log function
        ps = torch.exp(log_ps)

        # Get the top class of the output
        # Returns the k largest elements of the given input tensor along a given dimension.
        # Retruns value and index
        # https://pytorch.org/docs/stable/torch.html#torch.topk
        top_p, top_class = ps.topk(1, dim=1)

        # Compute how many classes are correct.
        equals = top_class == labels.view(*top_class.shape)

        # Calculate the mean (get the accuracy for this batch)
        # and add it to the running accuracy for this epoch
        accuracy += torch.mean(equals.type(torch.FloatTensor))

      # Revert model back to training mode
      model.train()

      # Get the average loss for the entire epoch
      train_losses.append(running_loss/len(trainloader))
      test_losses.append(test_loss/len(testloader))

      print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_losses[-1]),
              "Test Loss: {:.3f}.. ".format(test_losses[-1]),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
