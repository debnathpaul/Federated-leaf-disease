"""
Predict leaf disease from an image.
Usage: python predict.py <image_path>
Example: python predict.py test_leaf.jpg
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.cnn_model import LeafDiseaseNet

MODEL_PATH = 'model/leaf_disease_model.pth'

CLASS_NAMES = [
    'Tomato_healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Potato___Early_blight',   
    'Potato___Late_blight'      
]

CLASS_INFO = {
    'Tomato_healthy':        ('✅ Healthy',          'The tomato plant is healthy'),
    'Tomato_Bacterial_spot': ('🔴 Bacterial Spot',   'Bacterial spot disease detected'),
    'Tomato_Early_blight':   ('🟠 Early Blight',     'Early blight fungal disease detected'),
    'Tomato_Late_blight':    ('🟡 Late Blight',      'Late blight disease detected'),
    'Potato___Early_blight': ('🟣 Potato Early Blight', 'Potato early blight detected'),
    'Potato___Late_blight':  ('🟤 Potato Late Blight',  'Potato late blight detected'),
}

def predict(image_path):
    print("="*60)
    print("Leaf Disease Detection")
    print("="*60)

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found: '{MODEL_PATH}'")
        print("Please run: python save_model.py")
        return

    # Check image exists
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found: '{image_path}'")
        return

    # Load model
    print(f"\n[1] Loading model...")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model = LeafDiseaseNet(num_classes=6)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"    Model loaded!")

    # Load and preprocess image
    print(f"\n[2] Processing image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    print(f"    Image size: {image.size}")

    # Predict
    print(f"\n[3] Predicting...")
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_idx].item() * 100

    predicted_class = CLASS_NAMES[predicted_idx]
    emoji_name, bangla = CLASS_INFO[predicted_class]

    # Results
    print(f"\n{'='*60}")
    print(f"RESULT:")
    print(f"  Class    : {predicted_class}")
    print(f"  Status   : {emoji_name}")
    print(f"  Info     : {bangla}")
    print(f"  Confidence: {confidence:.2f}%")
    print(f"\nTop 3 Predictions:")
    top3 = probabilities.topk(3)
    for i, (prob, idx) in enumerate(zip(top3.values, top3.indices)):
        name = CLASS_NAMES[idx.item()]
        print(f"  {i+1}. {name}: {prob.item()*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Example: python predict.py leaf.jpg")
    else:
        predict(sys.argv[1])
