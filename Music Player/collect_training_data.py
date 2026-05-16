"""
Collect Custom Training Data for ASL Recognition
Use this to collect your own hand gesture images for fine-tuning
"""
import cv2
import os
from pathlib import Path
import time

class DataCollector:
    def __init__(self, output_dir="datasets/custom_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_letter(self, letter, num_images=50):
        """Collect images for a specific letter"""
        letter = letter.upper()
        letter_dir = self.output_dir / letter
        letter_dir.mkdir(exist_ok=True)
        
        # Get existing images
        existing = len(list(letter_dir.glob("*.jpg")))
        print(f"\nCollecting data for letter: {letter}")
        print(f"Existing images: {existing}")
        print(f"Target: {num_images} images")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Camera not found!")
            return False
        
        collected = 0
        print(f"\nPress 'c' to capture, 'q' to skip to next letter")
        print(f"Make sure your hand is in front of the camera\n")
        
        while collected < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Draw ROI rectangle
            roi_size = 200
            x_center, y_center = w // 2, h // 2
            x1, y1 = x_center - roi_size // 2, y_center - roi_size // 2
            x2, y2 = x_center + roi_size // 2, y_center + roi_size // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Letter: {letter}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(frame, f"Collected: {collected}/{num_images}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(f'Collecting {letter}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Save ROI
                roi = frame[y1:y2, x1:x2]
                img_path = letter_dir / f"{letter}_{collected}.jpg"
                cv2.imwrite(str(img_path), roi)
                collected += 1
                print(f"✓ Captured {collected}/{num_images}")
            
            elif key == ord('q'):
                print(f"Skipping {letter}")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
    
    def collect_all_letters(self, num_per_letter=20):
        """Collect data for all 26 letters"""
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        print("\n" + "="*60)
        print("ASL LETTER COLLECTION")
        print("="*60)
        print(f"\nYou will collect data for {len(letters)} letters")
        print(f"Each letter: {num_per_letter} images")
        print(f"Total images: {len(letters) * num_per_letter}")
        print("\nTips:")
        print("- Good lighting (front-facing light)")
        print("- Clear hand without background")
        print("- Different hand positions/angles")
        print("- Centered in the green box")
        
        input("\nPress Enter to start...")
        
        for i, letter in enumerate(letters, 1):
            print(f"\n[{i}/{len(letters)}] Starting {letter}...")
            self.collect_letter(letter, num_per_letter)
        
        print("\n✓ Data collection complete!")
        total_images = sum(len(list((self.output_dir / l).glob("*.jpg"))) 
                          for l in letters)
        print(f"Total images collected: {total_images}")
        print(f"Location: {self.output_dir}")

def main():
    collector = DataCollector()
    
    # Option 1: Collect specific letters
    print("Options:")
    print("1. Collect all letters")
    print("2. Collect specific letters")
    print("3. Quick test (5 images per letter)")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        collector.collect_all_letters(num_per_letter=20)
    elif choice == "2":
        letters = input("Enter letters (e.g., ABC): ").upper().strip()
        num = int(input("Images per letter: "))
        for letter in letters:
            collector.collect_letter(letter, num)
    elif choice == "3":
        collector.collect_all_letters(num_per_letter=5)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
