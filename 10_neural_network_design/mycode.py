import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 input image channel, 6 output channels, 5x5 square convolution kernel
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
'''
def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray
) -> np.ndarray:
    
    length_predictions = len(prediction)
    matrix = np.zeros((length_predictions, length_predictions), int)

    for i in range(len(target)):
        current_class = target[i]
        predicted_class = prediction[i]
        matrix[predicted_class][current_class] += 1
        
    return matrix'''

from sklearn.metrics import confusion_matrix


if __name__ == '__main__':
    net = Net()
    '''
    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    # Train the network
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
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    '''
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
    '''

    PATH = './cifar_net.pth'
    #torch.save(net.state_dict(), PATH)

    net.load_state_dict(torch.load(PATH))

    outputs = net(images)

    _, predicted = torch.max(outputs, 1)
    '''
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    '''

    # Let us look at how the network performs on the whole dataset.
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # what are the classes that performed well, and the classes that did not perform well

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    misclassified_by_class = {classname: 0 for classname in classes}
    total_classification = {classname: 0 for classname in classes}

    true_labels = [] 
    predicted_labels = [] 


    # again no gradients needed
    with torch.no_grad():
        
        
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)

            '''
            # ACCURACY

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
        
            # MISCLASSIFICATION RATE
        
            for label, prediction in zip(labels, predictions):
                if label != prediction:
                    misclassified_by_class[classes[label]] += 1
                total_classification[classes[label]] += 1
        '''
            #CONFUSION MATRIX

            true_labels.extend(labels.numpy())  
            predicted_labels.extend(predicted.numpy()) 

    # Calculate the confusion matrix
    confusion = confusion_matrix(true_labels, predicted_labels)

    '''
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    misclassification_rates = []
    class_names = []

    for classname, misclassification_count in misclassified_by_class.items():
        misclassification_rate_by_class = misclassification_count / total_classification[classname]
        misclassification_rates.append(misclassification_rate_by_class)
        class_names.append(classname)

    # Plot the misclassification rate for each class
    plt.scatter(class_names, [rate * 100 for rate in misclassification_rates], label='Misclassification Rate')
    plt.xlabel('Class')
    plt.ylabel('Misclassification Rate (%)')
    plt.title('Misclassification Rate by Class')
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    #
    print("Confusion Matrix: \n", confusion)