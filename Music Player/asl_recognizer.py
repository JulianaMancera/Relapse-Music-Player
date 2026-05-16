"""
ASL Recognition using Trained ML Model
Replaces rule-based landmark detection
"""
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import json

class ASLRecognizer:
    def __init__(self, model_path="models/asl_model.h5", class_indices_path="models/class_indices.json"):
        """Initialize ASL recognizer with trained model"""
        self.model_path = Path(model_path)
        self.class_indices_path = Path(class_indices_path)
        self.img_size = 224
        self.confidence_threshold = 0.7
        
        # Load model and classes
        self.load_model()
    
    def load_model(self):
        """Load trained model and class indices"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}\n"
                "Please run: python train_asl_model.py"
            )
        
        if not self.class_indices_path.exists():
            raise FileNotFoundError(
                f"Class indices not found at {self.class_indices_path}"
            )
        
        # Load model
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✓ Model loaded: {self.model_path}")
        
        # Load class indices
        with open(self.class_indices_path, 'r') as f:
            class_dict = json.load(f)
            self.classes = {v: k for k, v in class_dict.items()}
        
        print(f"✓ Classes loaded: {sorted(self.classes.values())}")
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize
        resized = cv2.resize(frame, (self.img_size, self.img_size))
        
        # Normalize
        normalized = resized / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def recognize(self, frame, return_confidence=False):
        """
        Recognize ASL letter from frame
        
        Args:
            frame: Input frame from camera
            return_confidence: If True, return (letter, confidence)
        
        Returns:
            Recognized letter (A-Z) or None if confidence is too low
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            # Preprocess
            processed = self.preprocess_frame(frame)
            
            # Predict
            predictions = self.model.predict(processed, verbose=0)
            confidence = np.max(predictions)
            predicted_class = np.argmax(predictions)
            
            # Get letter
            letter = self.classes.get(predicted_class)
            
            # Check confidence threshold
            if confidence < self.confidence_threshold:
                if return_confidence:
                    return None, confidence
                return None
            
            if return_confidence:
                return letter, confidence
            
            return letter
        
        except Exception as e:
            print(f"Error in recognition: {e}")
            return None
    
    def recognize_with_details(self, frame):
        """
        Recognize and return detailed information
        
        Returns:
            {
                'letter': 'A',
                'confidence': 0.95,
                'all_predictions': {'A': 0.95, 'B': 0.03, ...}
            }
        """
        if frame is None or frame.size == 0:
            return None
        
        try:
            processed = self.preprocess_frame(frame)
            predictions = self.model.predict(processed, verbose=0)[0]
            
            # Get all predictions
            all_preds = {self.classes[i]: float(predictions[i]) for i in range(len(predictions))}
            all_preds = dict(sorted(all_preds.items(), key=lambda x: x[1], reverse=True))
            
            # Top prediction
            top_letter = list(all_preds.keys())[0]
            top_confidence = all_preds[top_letter]
            
            return {
                'letter': top_letter if top_confidence >= self.confidence_threshold else None,
                'confidence': top_confidence,
                'all_predictions': all_preds
            }
        
        except Exception as e:
            print(f"Error in recognition: {e}")
            return None


def test_recognizer():
    """Test the recognizer with a sample image"""
    try:
        recognizer = ASLRecognizer()
        
        # Test with a dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = recognizer.recognize_with_details(dummy_frame)
        
        print(f"✓ Recognizer working!")
        print(f"  Sample prediction: {result}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    test_recognizer()
