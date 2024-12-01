import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageComparator:
    def __init__(self):
        try:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.model.eval()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ResNet50 model: {str(e)}")

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract feature embedding from an image.
        
        Args:
            image (PIL.Image.Image): Input image
            
        Returns:
            np.ndarray: Feature embedding
        """
        if not isinstance(image, Image.Image):
            raise TypeError("Input must be a PIL Image object")
            
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model(image_tensor)
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate image embedding: {str(e)}")

    def calculate_similarity(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Calculate similarity score between two images.
        
        Args:
            image1 (PIL.Image.Image): First image
            image2 (PIL.Image.Image): Second image
            
        Returns:
            float: Similarity score between 0 and 100
        """
        try:
            embedding1 = self._get_image_embedding(image1)
            embedding2 = self._get_image_embedding(image2)
            similarity = np.dot(embedding1, embedding2)
            similarity_score = round(((similarity + 1) / 2) * 100, 2)
            return similarity_score
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate image similarity: {str(e)}")

    def __call__(self, image1: Image.Image, image2: Image.Image) -> float:
        return self.calculate_similarity(image1, image2)
