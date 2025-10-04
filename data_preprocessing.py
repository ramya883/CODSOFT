import os
import string
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications import ResNet50, VGG16
from keras.applications.resnet50 import preprocess_input as resnet_preprocess
from keras.applications.vgg16 import preprocess_input as vgg_preprocess
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from collections import defaultdict

class DataPreprocessor:
    def __init__(self, captions_file, images_folder, max_caption_length=20, vocab_size=10000, cnn_model_type="resnet50"):
        self.captions_file = captions_file
        self.images_folder = images_folder
        self.max_caption_length = max_caption_length
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.cnn_model = None
        self.cnn_model_type = cnn_model_type.lower()  # "resnet50" or "vgg16"
        
        # Feature dimensions based on model type
        self.feature_dim = 2048 if self.cnn_model_type == "resnet50" else 4096  # VGG16 has 4096 features
        
    def load_captions(self):
        """Load captions from file and organize by image"""
        captions_dict = defaultdict(list)
        
        try:
            with open(self.captions_file, "r", encoding='utf-8') as f:
                next(f)  # Skip header
                for line in f:
                    line = line.strip()
                    if len(line) < 2:
                        continue
                    
                    # Split by comma, but handle cases where caption contains commas
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        img_name, caption = parts
                        img_name = img_name.strip()
                        caption = caption.strip()
                        captions_dict[img_name].append(caption)
                        
        except Exception as e:
            print(f"Error loading captions: {e}")
            return {}
            
        return dict(captions_dict)
    
    def clean_captions(self, captions_dict):
        """Clean and preprocess captions"""
        table = str.maketrans("", "", string.punctuation)
        cleaned = {}
        
        for img_id, captions in captions_dict.items():
            cleaned[img_id] = []
            for caption in captions:
                # Convert to lowercase and remove punctuation
                caption = caption.lower().translate(table).strip()
                # Add start and end tokens
                caption = "<start> " + caption + " <end>"
                cleaned[img_id].append(caption)
                
        return cleaned
    
    def create_tokenizer(self, captions_dict):
        """Create tokenizer for captions"""
        all_captions = []
        for captions in captions_dict.values():
            all_captions.extend(captions)
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<unk>")
        self.tokenizer.fit_on_texts(all_captions)
        
        # Add special tokens if not present
        if "<start>" not in self.tokenizer.word_index:
            self.tokenizer.word_index["<start>"] = len(self.tokenizer.word_index) + 1
        if "<end>" not in self.tokenizer.word_index:
            self.tokenizer.word_index["<end>"] = len(self.tokenizer.word_index) + 1
            
        return self.tokenizer
    
    def init_cnn_model(self):
        """Initialize CNN model for feature extraction"""
        if self.cnn_model_type == "resnet50":
            self.cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        elif self.cnn_model_type == "vgg16":
            self.cnn_model = VGG16(weights="imagenet", include_top=False, pooling="avg")
        else:
            raise ValueError(f"Unsupported CNN model type: {self.cnn_model_type}. Use 'resnet50' or 'vgg16'.")
        return self.cnn_model
    
    def extract_image_features(self, image_path):
        """Extract features from image using CNN model"""
        if self.cnn_model is None:
            self.init_cnn_model()
            
        try:
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply appropriate preprocessing based on model type
            if self.cnn_model_type == "resnet50":
                img_array = resnet_preprocess(img_array)
            else:  # vgg16
                img_array = vgg_preprocess(img_array)
            
            features = self.cnn_model.predict(img_array, verbose=0)
            return features.flatten()
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def create_sequences(self, captions_dict):
        """Create input-output sequences for training"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call create_tokenizer first.")
            
        image_names = []
        input_sequences = []
        target_sequences = []
        
        for img_id, captions in captions_dict.items():
            img_path = os.path.join(self.images_folder, img_id)
            
            # Check if image exists
            if not os.path.exists(img_path):
                continue
                
            for caption in captions:
                # Tokenize caption
                sequence = self.tokenizer.texts_to_sequences([caption])[0]
                
                # Create input-output pairs
                for i in range(1, len(sequence)):
                    input_seq = sequence[:i]
                    target_seq = sequence[1:i+1]  # Shift target by 1 position
                    
                    # Pad sequences
                    input_seq = pad_sequences([input_seq], maxlen=self.max_caption_length, padding='post')[0]
                    target_seq = pad_sequences([target_seq], maxlen=self.max_caption_length, padding='post')[0]
                    
                    image_names.append(img_id)
                    input_sequences.append(input_seq)
                    target_sequences.append(target_seq)
                    
        return image_names, input_sequences, target_sequences
    
    def preprocess_data(self, test_size=0.2, val_size=0.1, max_images=None):
        """Complete preprocessing pipeline
        Args:
            test_size: fraction for test split
            val_size: fraction for validation split
            max_images: optional int to limit the number of distinct images processed (for quick runs)
        """
        print("Loading captions...")
        captions_dict = self.load_captions()
        print(f"Loaded captions for {len(captions_dict)} images")
        
        print("Cleaning captions...")
        cleaned_captions = self.clean_captions(captions_dict)
        
        # Optionally limit number of images for faster experiments
        if max_images is not None:
            all_img_ids = list(cleaned_captions.keys())[:max_images]
            cleaned_captions = {k: cleaned_captions[k] for k in all_img_ids}
            print(f"Limiting to {len(cleaned_captions)} images for preprocessing")
        
        print("Creating tokenizer...")
        self.create_tokenizer(cleaned_captions)
        
        print("Creating sequences...")
        image_names, input_sequences, target_sequences = self.create_sequences(cleaned_captions)
        
        print(f"Created {len(input_sequences)} training sequences")
        
        # Extract features for images
        print("Extracting image features...")
        image_features = {}
        
        for img_name in set(image_names):
            img_path = os.path.join(self.images_folder, img_name)
            features = self.extract_image_features(img_path)
            if features is not None:
                image_features[img_name] = features
        
        print(f"Extracted features for {len(image_features)} images")
        
        # Filter sequences to only include images with features
        valid_indices = [i for i, img_name in enumerate(image_names) if img_name in image_features]
        
        image_names = [image_names[i] for i in valid_indices]
        input_sequences = [input_sequences[i] for i in valid_indices]
        target_sequences = [target_sequences[i] for i in valid_indices]
        
        # Convert to numpy arrays
        input_sequences = np.array(input_sequences)
        target_sequences = np.array(target_sequences)
        
        # Split data
        X_train, X_temp, y_train, y_temp, img_train, img_temp = train_test_split(
            input_sequences, target_sequences, image_names, test_size=test_size + val_size, random_state=42
        )
        
        if val_size > 0:
            X_val, X_test, y_val, y_test, img_val, img_test = train_test_split(
                X_temp, y_temp, img_temp, test_size=val_size/(test_size + val_size), random_state=42
            )
        else:
            X_val, y_val, img_val = None, None, None
            X_test, y_test, img_test = X_temp, y_temp, img_temp
        
        return {
            'train': (X_train, y_train, img_train),
            'val': (X_val, y_val, img_val),
            'test': (X_test, y_test, img_test),
            'image_features': image_features,
            'tokenizer': self.tokenizer,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'max_length': self.max_caption_length,
            'captions_dict': cleaned_captions
        }
    
    def save_preprocessed_data(self, data, save_path="preprocessed_data.pkl"):
        """Save preprocessed data to file"""
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Preprocessed data saved to {save_path}")
    
    def load_preprocessed_data(self, load_path="preprocessed_data.pkl"):
        """Load preprocessed data from file"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Preprocessed data loaded from {load_path}")
        return data

if __name__ == "__main__":
    # Example usage
    captions_file = "captions/captions.txt"
    images_folder = "images"
    
    preprocessor = DataPreprocessor(captions_file, images_folder)
    data = preprocessor.preprocess_data()
    preprocessor.save_preprocessed_data(data)
    
    print("Preprocessing complete!")
    print(f"Training samples: {len(data['train'][0])}")
    print(f"Vocabulary size: {data['vocab_size']}")