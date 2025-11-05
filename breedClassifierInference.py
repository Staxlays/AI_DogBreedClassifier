#abari
#AI Dog Breed Classifier - Inference module

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from PIL import Image
import sys


#Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Transform the input image to match the training images' format
inferenceTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#To get the breed names I need to load the training dataset again
trainData = datasets.ImageFolder('data/train')
breedNames = trainData.classes
numBreeds = len(breedNames)
print(f"Number of breeds: {numBreeds}")

#Load the trained model and the breed names
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, numBreeds)

#Used the saved model to import the trained weights
#Adding the same check from the breedClassifierTraining.py file
try:
    model.load_state_dict(torch.load("dogBreedClassifier.pth", map_location=device))
except:
    print("Error: Could not find the saved model, please ensure the file is saved to the same folder as this evaluation script and it is name \'dog_breed_resnet50.pth\'")
    exit()

#Mount the model and set it to eval mode
model = model.to(device)
model.eval()

#PREDICTION TIMEEEEE
def predictBreed(imagePath, topk=3):
    #Load the image and transform it to conform to the model's input format
    try:
        image = Image.open(imagePath).convert('RGB')
    except:
        print("Error: Could not open the image, please ensure the file path is correct and the image is a valid format (JPG, PNG, etc.)")
        return
    
    #Apply the transformations for the model's input
    tensoredImage = inferenceTransform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensoredImage)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        topProbabilities, topIndices = probabilities.topk(topk, dim=1)

    #Record the results
    results = []
    for probability, indice in zip(topProbabilities[0], topIndices[0]):
        breedName = breedNames[indice.item()]
        confidence = probability.item()
        results.append((breedName, confidence))
    
    return results

#Main function to pass in an image path from the comman line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Proper Usage: python breedClassifierInference.py <image_path>")
        sys.exit()
    imagePath=sys.argv[1]
    predictions = predictBreed(imagePath, topk=3)

    if predictions == None:
        print("No predictions made. Exiting.")
        sys.exit()
    else:
        print("\nTop Predictions:\n")
        for breed, confidence in predictions:
            print(f"Breed: {breed}, Confidence: {confidence:.2f}")
    
