"""
Train ASL Recognition Model using Transfer Learning
This replaces the rule-based landmark detection with a ML model
"""
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from pathlib import Path
import json


class LegacyBatchNormalization(keras.layers.BatchNormalization):
    """Compatibility shim for older Keras BatchNormalization configs."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('renorm', None)
        config.pop('renorm_clipping', None)
        config.pop('renorm_momentum', None)
        return cls(**config)


class LegacyDense(keras.layers.Dense):
    """Compatibility shim for Dense configs saved by newer Keras versions."""

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop('quantization_config', None)
        return cls(**config)

class StateCallback(Callback):
    def __init__(self, state_file, phase):
        self.state_file = state_file
        self.phase = phase

    def on_epoch_end(self, epoch, _logs=None):
        with open(self.state_file, "w") as f:
            json.dump({"phase": self.phase, "epoch": epoch + 1}, f)

class ASLModelTrainer:
    def __init__(self, dataset_dir="datasets/asl-alphabet", model_dir="models"):
        self.dataset_dir = Path(dataset_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Kaggle extracts with double nesting: asl_alphabet_train/asl_alphabet_train/A/
        _train_outer = self.dataset_dir / "asl_alphabet_train"
        _train_inner = _train_outer / "asl_alphabet_train"
        self.train_dir = _train_inner if _train_inner.exists() else _train_outer

        _test_outer = self.dataset_dir / "asl_alphabet_test"
        _test_inner = _test_outer / "asl_alphabet_test"
        self.test_dir = _test_inner if _test_inner.exists() else _test_outer
        
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
        
        # Test data (if available and has class subdirectories)
        _test_has_classes = self.test_dir.exists() and any(
            d.is_dir() for d in self.test_dir.iterdir()
        )
        if _test_has_classes:
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
        """Train the model (resumable from checkpoint)"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)

        # Check for resume state
        state_file = self.model_dir / "training_state.json"
        checkpoint_path = self.model_dir / "best_model.h5"
        resume_phase = 1
        resume_epoch = 0

        if state_file.exists():
            with open(state_file, "r") as f:
                state = json.load(f)
            resume_phase = state["phase"]
            resume_epoch = state["epoch"]
            print(f"\n✓ Resuming from Phase {resume_phase}, Epoch {resume_epoch}")
        elif checkpoint_path.exists():
            # Checkpoint exists but state file is missing — Phase 1 was complete
            resume_phase = 2
            resume_epoch = 10
            print(f"\n⚠ State file missing but checkpoint found — assuming Phase 1 complete, starting Phase 2")

        # Verify dataset
        if not self.verify_dataset():
            return False

        # Prepare data
        train_gen, val_gen, test_gen = self.prepare_data()

        # Build model
        model, base_model = self.build_model(num_classes=train_gen.num_classes)

        # Load checkpoint whenever it exists and we have a resume point
        if checkpoint_path.exists() and resume_epoch > 0:
            print(f"✓ Loading checkpoint from best_model.h5")
            model = keras.models.load_model(
                checkpoint_path,
                custom_objects={
                    'BatchNormalization': LegacyBatchNormalization,
                    'Dense': LegacyDense,
                },
                compile=False,
            )
            # Recompile with a fresh optimizer — Keras 3 stale optimizer causes
            # "Unknown variable" crash when resuming from .h5 checkpoints
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            # Re-derive base_model from the loaded model so Phase 2 unfreezes the right layers
            for layer in model.layers:
                if 'mobilenet' in layer.name.lower():
                    base_model = layer
                    break

        # Callbacks (Phase 1)
        callbacks_phase1 = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=False,
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
            ),
            StateCallback(state_file, phase=1)
        ]

        # Phase 1: Train with frozen base (faster)
        if resume_phase == 1:
            print("\n📊 Phase 1: Training top layers (frozen backbone)...")
            _ = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=10,
                initial_epoch=min(resume_epoch, 10),
                callbacks=callbacks_phase1,
                verbose=1
            )
            resume_phase = 2
            resume_epoch = 10
            # Write state so a restart skips Phase 1
            with open(state_file, "w") as f:
                json.dump({"phase": 2, "epoch": 10}, f)

        # Phase 2: Fine-tune (unfreeze some layers)
        if resume_phase == 2:
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

            callbacks_phase2 = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=False,
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
                ),
                StateCallback(state_file, phase=2)
            ]

            _ = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                initial_epoch=max(resume_epoch, 10),
                callbacks=callbacks_phase2,
                verbose=1
            )
        
        # Evaluate
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        _, val_acc = model.evaluate(val_gen)
        print(f"Validation Accuracy: {val_acc*100:.2f}%")
        
        if test_gen:
            _, test_acc = model.evaluate(test_gen)
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
