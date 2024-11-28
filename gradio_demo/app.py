import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

class CatDogClassifierGradio:
    def __init__(self, best_model_path="model_storage/best_model_checkpoint.txt"):

        # Initialize the variable to store the second line
        model_path = ""

        # Read the file and fetch the second line
        with open(best_model_path, "r") as file:
            lines = file.readlines()
            if len(lines) > 1:  # Check if the file has at least two lines
                model_path = lines[1].strip()  # Strip removes any trailing newline or spaces

        # Print or use the second line as needed
        print("model_path :", model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the traced model
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model.eval()
        # test comment
        # test comment 2

        # Define the same transforms used during training/testing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Class labels
        self.labels = ['Cat', 'Dog']

    @torch.no_grad()
    def predict(self, image):
        if image is None:
            return None
        
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        output = self.model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Create prediction dictionary
        return {
            self.labels[idx]: float(prob)
            for idx, prob in enumerate(probabilities)
        }

# Create classifier instance
classifier = CatDogClassifierGradio()

# Create Gradio interface
demo = gr.Interface(
    fn=classifier.predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="Cat vs Dog Classifier",
    description="Upload an image to classify whether it's a cat or a dog \n Note: The model accuracy is around 55% only not trained well as its for testing ci/cd pipelines",
    examples=[
        ["examples/cat.jpg"],
        ["examples/dog.jpg"]
    ]
)

if __name__ == "__main__":
    demo.launch(share=True) 
