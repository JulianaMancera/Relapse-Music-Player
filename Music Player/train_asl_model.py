"""
Train ASL Recognition Model using Transfer Learning
This replaces the rule-based landmark detection with a ML model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path
import json

class ASLModelTrainer:
    def __init__(self, dataset_dir="datasets/asl-alphabet", model_dir="models"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.train_dir = self.dataset_dir / "asl_alphabet_train"
        self.test_dir = self.dataset_dir / "asl_alphabet_test"
        
        # Image parameters
        self.img_size = 224
        self.batch_size = 32
        
    def verify_dataset(self):
        """Check if dataset exists"""
        if not self.train_dir.exists():
            print(f"❌ Dataset not found at {self.train_dir}")
            print("   Run: python download_dataset.py")
            return False
        
        classes = [d.name for d in self.train_dir.iterdir() if d.is_dir()]
        print(f"✓ Found {len(classes)} letter classes: {sorted(classes)[:10]}...")
        
        # Count images
        total_images = sum(len(list(f.glob("*"))) for f in self.train_dir.iterdir() if f.is_dir())
        print(f"✓ Total training images: {total_images}")
        
        return True
    
    def prepare_data(self):
        """Create data generators with augmentation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Training data
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )
        
        # Validation data
        val_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )
        
        # Test data (if available)
        if self.test_dir.exists():
            test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(self.img_size, self.img_size),
                batch_size=self.batch_size,
                class_mode='categorical'
            )
        else:
            test_generator = None
        
        # Save class indices
        class_indices = train_generator.class_indices
        with open(self.model_dir / "class_indices.json", "w") as f:
            json.dump(class_indices, f)
        
        print(f"✓ Classes: {sorted(class_indices.keys())}")
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        
        return train_generator, val_generator, test_generator
    
    def build_model(self, num_classes=26):
        """Build model using MobileNetV2 transfer learning"""
        # Load pretrained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(self.img_size, self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom layers
        inputs = Input(shape=(self.img_size, self.img_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✓ Model created with {model.count_params():,} parameters")
        return model, base_model
    
    def train(self, epochs=20):
        """Train the model"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Verify dataset
        if not self.verify_dataset():
            return False
        
        # Prepare data
        train_gen, val_gen, test_gen = self.prepare_data()
        
        # Build model
        model, base_model = self.build_model(num_classes=train_gen.num_classes)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_dir / "best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Phase 1: Train with frozen base (faster)
        print("\n📊 Phase 1: Training top layers (frozen backbone)...")
        history1 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tune (unfreeze some layers)
        print("\n📊 Phase 2: Fine-tuning with unfrozen backbone...")
        base_model.trainable = True
        
        # Unfreeze last 30 layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs - 10,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        val_loss, val_acc = model.evaluate(val_gen)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        if test_gen:
            test_loss, test_acc = model.evaluate(test_gen)
            print(f"Test Accuracy: {test_acc*100:.2f}%")
        
        # Save model
        model.save(self.model_dir / "asl_model.h5")
        print(f"\n✓ Model saved to: {self.model_dir / 'asl_model.h5'}")
        
        return True

def main():
    trainer = ASLModelTrainer()
    trainer.train(epochs=20)

if __name__ == "__main__":
    main()
