#abari
#AI Dog Breed Classifier - Model Evaluation

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd


#Setting up the device used for evaluation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#The data used to evaluate the model needs to get standardized the same way as the training data
testTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #No need to flip or rotate evaluation images
])

#debug print statement, had issues with loading the test dataset
print("Loading test dataset...")
#Loading the dataset for model evaluation
testData = datasets.ImageFolder('data/test', transform=testTransform)
testLoader = torch.utils.data.DataLoader(testData, batch_size=32, shuffle=False)

#Getting the number of breeds and names for each breed
numBreeds = len(testData.classes)
breedNames = testData.classes
print(f"Number of breeds: {numBreeds}")
print(f"Number of test images: {len(testData)}")

#Loading in the saved model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, numBreeds)

#Need to load the model saved from the training session
#Putting in a safety net in case the training file is not found
try:
    model.load_state_dict(torch.load("dogBreedClassifier.pth", map_location=device))
except:
    print("Error: Could not find the saved model, please ensure the file is saved to the same folder as this evaluation script and it is name \'dog_breed_resnet50.pth\'")
    exit()

model = model.to(device)
model.eval()



#EVALUATION TIMEEEE
print("Beginning evaluation process!")
all_preds = []
all_labels = []

start_time = time.time()

with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in testLoader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#Recording and printing out the final statistics for the evaluation
evalAcc = correct / total
endTime = time.time()
timeElapsed = endTime - start_time
print(f"Evaluation completed in {timeElapsed//60:.0f}m {timeElapsed%60:.0f}s")
print(f"Evaluation Accuracy: {evalAcc:.4f}")




#
#REPORTING TIMEEEE
#

#Creating confusion matrix first
#Normalizing the confusion matrix, it was way too hard to read without it
print("Generating confusion matrix...")

cm = confusion_matrix(all_labels, all_preds, normalize='true')
fig, ax = plt.subplots(figsize=(30, 30))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=breedNames)
disp.plot(cmap=plt.cm.Greens, xticks_rotation = 90, ax=ax, colorbar=True)
plt.title("Normalized Confusion Matrix for Dog Breed Classifier")
plt.tight_layout()

plt.savefig("confusionMatrix.png")
plt.close()

print("Confusion matrix saved as \'confusionMatrix.png\'")

#Sorting best and worst performing breeds then saving the results to a CSV file
print("Creating per-breed accuracy CSV...")

breedAcc = {}
for i, breed in enumerate(breedNames):
    breedTotal = np.sum(np.array(all_labels) == i)
    breedCorrect = np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
    individualBreedAcc = breedCorrect / breedTotal if breedTotal > 0 else 0
    breedAcc[breed] = individualBreedAcc

#Sort the breed performance for clarity
breedSorted = sorted(breedAcc.items(), key=lambda item:item[1], reverse=True)

print("\nTop 5 Breeds Accurately Classified:")
for breed, acc in breedSorted[:5]:
    print(f"{breed}: {acc:.2f}%")

print("\nBottom 5 Breeds Accurately Classified:")
for breed, acc in breedSorted[-5:]:
    print(f"{breed}: {acc:.2f}%")

df=pd.DataFrame(breedSorted, columns=["Breed", "Accuracy"])
df.to_csv("perBreedAccuracy.csv", index=False)
print("Per-breed accuracy saved as \'perBreedAccuracy.csv\'")


#Creating and saving a bar chart for breed accuracy
print("Generating breed accuracy bar chart...")

plt.figure(figsize=(14, 40))
plt.barh(df["Breed"], df["Accuracy"], color="forestgreen")
plt.xlabel("Accuracy", fontsize=14)
plt.ylabel("Breed", fontsize=14)
plt.title("Dog Breed Classification Accuracy by Breed", fontsize=16)
plt.gca().invert_yaxis()
plt.yticks(fontsize=10)
plt.subplots_adjust(left=.4, right=.95, top=.97, bottom=.03)

plt.savefig("breedAccuracyBarChart.png")
plt.close()
print("Breed accuracy bar chart saved as \'breedAccuracyBarChart.png\'")

print("Evaluation process complete! Please check for the results in BreedAccuracyBarChart.png, confusion_matrix.png, and perBreedAccuracy.csv files saved to the working directory.")