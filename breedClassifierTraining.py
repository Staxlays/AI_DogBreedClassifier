#abari
#AI Dog Breed Classifier - Training Script

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import time
import matplotlib.pyplot as plt




#checking which device the model will use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#Transformations for the training and validation sets
#All the images must be resized to 224x224

trainingTransformer = transforms.Compose([
    #Standardizing the images to be the same size
    transforms.Resize((224, 224)),
    #Converting the image files to tensors
    transforms.ToTensor(),
    #Normalizing the images to be in the range of 0-1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #Adding a random image flipper to try and make the model more robust
    transforms.RandomHorizontalFlip()
])

validationTransformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #No random flip for the validation set
])

#Importing the datasets for training and validation
trainingData = datasets.ImageFolder('data/train', transform=trainingTransformer)
validationData = datasets.ImageFolder('data/val', transform=validationTransformer)

trainingLoader = torch.utils.data.DataLoader(trainingData, batch_size=32, shuffle=True)
validationLoader = torch.utils.data.DataLoader(validationData, batch_size=32, shuffle=False)

#printing out number of breeds and images in each set
numClasses = len(trainingData.classes)
numTrainingImages = len(trainingData)
numValidationImages = len(validationData)
print(f"Number of breeds classified: {numClasses}\nNumber of training images: {numTrainingImages}\nNumber of validation images: {numValidationImages}")

#Time to train the model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

#Initially freezing all of the CNN layers
for param in model.parameters():
    param.requires_grad = False

#Apparently its good to match the final layer to the number of classes
model.fc = nn.Linear(model.fc.in_features, numClasses)

#Loading the model to the device
model = model.to(device)


#Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

#BEGIN THE TRAINING PROCESS
def breedClassifierTraining(model, train_loader, val_loader, criterion, optimizer, epochs=15):
    #keep track of how long the training takes
    startTime = time.time()
    #best accuracy and loss results
    bestTrainingAcc = 0.0
    bestValidationAcc = 0.0
    bestTrainingLoss = float('inf')
    bestValidationLoss = float('inf')

    # Matplotlib plot variables for visualizing training results (one entry per epoch)
    trainLosses = []
    valLosses = []
    trainAccs = []
    valAccs = []

    #Number of training cycles determined by number of epochs set
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("--------------------")
        model.train()
        trainLoss = 0.0
        trainCorrects = 0


        for inputs, labels in trainingLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * inputs.size(0)
            trainCorrects += torch.sum(preds == labels.data)
        
        #Record the statistics for the current epoch
        #Had to cast the trainAcc and trainLoss to a Python float in order to avoid a matplotlib error
        trainLoss = float(trainLoss / len(train_loader.dataset))
        trainAcc = float(trainCorrects.double() / len(train_loader.dataset))
        print(f"Training Loss: {trainLoss:.4f} Acc: {trainAcc:.4f}")

        #Save the best training results
        if trainAcc > bestTrainingAcc:
            bestTrainingAcc = trainAcc
        if trainLoss < bestTrainingLoss:
            bestTrainingLoss = trainLoss

        #append the result to the matplotlib variables
        trainLosses.append(trainLoss)
        trainAccs.append(trainAcc)

        #Validation phase for the epoch
        model.eval()
        valLoss = 0.0
        valCorrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                #Get the right and wrong prediction numbers
                valLoss += loss.item() * inputs.size(0)
                valCorrects += torch.sum(preds == labels.data)

        #Getting the right and wrong predictions as a percentage
        #Had to cast the valAcc and valLoss to a Python float in order to avoid a matplotlib error
        valLoss = float(valLoss / len(val_loader.dataset))
        valAcc = float(valCorrects.double() / len(val_loader.dataset))
        print(f"Validation Loss: {valLoss:.4f} Acc: {valAcc:.4f}")

        #Save the best validation results
        if valAcc > bestValidationAcc:
            bestValidationAcc = valAcc
        if valLoss < bestValidationLoss:
            bestValidationLoss = valLoss

        #Append the results to the matplotlib variables
        valLosses.append(valLoss)
        valAccs.append(valAcc)

    
    #Recording and printing the final statistics for training session
    completedTime = time.time()
    timeElapsed = completedTime - startTime
    print(f"Training complete in {timeElapsed//60:.0f}m {timeElapsed%60:.0f}s")
    print(f"Best Training Acc: {bestTrainingAcc:.4f} Best Training Loss: {bestTrainingLoss:.4f}")
    print(f"Best Validation Acc: {bestValidationAcc:.4f} Best Validation Loss: {bestValidationLoss:.4f}")
    
    
    #Debug print statements for the matplotlib variables
    """
    print(f"Train Losses: {trainLosses}")
    print(f"Val Losses: {valLosses}")
    print(f"Train Accs: {trainAccs}")
    print(f"Val Accs: {valAccs}")
    """

    #Creating the matplotlib plots for training and validation results
    epochsRange = range(1, epochs+1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochsRange, trainLosses, linewidth = 2, marker = 'x', label='Training Loss')
    plt.plot(epochsRange, valLosses, linewidth = 2, marker = 'o', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("lossPlot.png")
    plt.close()
    print("Training and Validation Loss Plot Saved as 'loss_plot.png'")

    plt.figure(figsize=(10, 5))
    plt.plot(epochsRange, trainAccs, linewidth = 2, marker = 'x',label='Training Accuracy')
    plt.plot(epochsRange, valAccs, linewidth = 2, marker = 'o', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig("accuracyPlot.png")
    plt.close()
    print("Training and Validation Accuracy Plot Saved as 'accuracy_plot.png'")
    
    return model

#Creating the model and calling the training function
dogBreedModel = breedClassifierTraining(model, trainingLoader, validationLoader, criterion, optimizer, epochs=15)

#Saving the trained model
torch.save(dogBreedModel.state_dict(), 'dogBreedClassifier.pth')
print("Model saved as dogBreedClassifier.pth")