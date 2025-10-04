import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, add, RepeatVector, TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import pickle
import os

class ImageCaptioningModel:
    def __init__(self, vocab_size, max_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        
    def build_model(self):
        """Build the image captioning model"""
        # Image feature input
        image_input = Input(shape=(2048,), name='image_input')
        
        # Text input
        text_input = Input(shape=(self.max_length,), name='text_input')
        
        # Image feature processing
        image_dense = Dense(self.embedding_dim, activation='relu')(image_input)
        image_dropout = Dropout(0.5)(image_dense)
        
        # Text embedding
        text_embedding = Embedding(self.vocab_size, self.embedding_dim)(text_input)
        text_dropout = Dropout(0.5)(text_embedding)
        
        # LSTM layers
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(text_dropout)
        lstm1_dropout = Dropout(0.5)(lstm1)
        
        lstm2 = LSTM(self.lstm_units, return_sequences=True)(lstm1_dropout)
        lstm2_dropout = Dropout(0.5)(lstm2)
        
        # Combine image and text features
        # First, project image features to match LSTM dimensions
        image_projected = Dense(self.lstm_units, activation='relu')(image_dropout)
        
        # Repeat image features to match sequence length
        image_repeated = RepeatVector(self.max_length)(image_projected)
        
        # Concatenate image and text features
        combined = add([image_repeated, lstm2_dropout])
        
        # Final dense layer - predict next token at each position
        output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(combined)
        
        # Create model
        self.model = Model(inputs=[image_input, text_input], outputs=output)
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def get_model_summary(self):
        """Print model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def prepare_data_for_training(self, data_dict):
        """Prepare data for training"""
        X_train_seq, y_train, img_train = data_dict['train']
        X_val_seq, y_val, img_val = data_dict['val']
        
        image_features = data_dict['image_features']
        
        # Get image features for training
        train_features = np.array([image_features[img_name] for img_name in img_train])
        val_features = np.array([image_features[img_name] for img_name in img_val]) if X_val_seq is not None else None
        
        # Ensure y_train and y_val are 2D arrays for sparse_categorical_crossentropy
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and len(y_val.shape) == 1:
            y_val = y_val.reshape(-1, 1)
        
        return {
            'train': (train_features, X_train_seq, y_train),
            'val': (val_features, X_val_seq, y_val)
        }
    
    def train(self, data_dict, batch_size=64, epochs=50, model_save_path='image_captioning_model.h5'):
        """Train the model"""
        if self.model is None:
            self.build_model()
        
        # Prepare data
        prepared_data = self.prepare_data_for_training(data_dict)
        
        train_features, X_train_seq, y_train = prepared_data['train']
        val_features, X_val_seq, y_val = prepared_data['val']
        
        print(f"Training samples: {len(train_features)}")
        if val_features is not None:
            print(f"Validation samples: {len(val_features)}")
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-7
            )
        ]
        
        # Train model
        if val_features is not None:
            history = self.model.fit(
                [train_features, X_train_seq],
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([val_features, X_val_seq], y_val),
                callbacks=callbacks,
                verbose=1
            )
        else:
            history = self.model.fit(
                [train_features, X_train_seq],
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            )
        
        return history
    
    def save_model(self, model_path='image_captioning_model.h5', tokenizer_path='tokenizer.pkl'):
        """Save model and tokenizer"""
        if self.model is None:
            raise ValueError("Model not built yet")
            
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        if hasattr(self, 'tokenizer'):
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path='image_captioning_model.h5', tokenizer_path='tokenizer.pkl'):
        """Load model and tokenizer"""
        # Load model with custom objects if needed
        try:
            self.model = tf.keras.models.load_model(model_path)
        except ValueError as e:
            if "Unknown layer" in str(e) or "NotEqual" in str(e):
                # Try loading with custom objects scope
                from tensorflow.keras.utils import custom_object_scope
                from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, RepeatVector, TimeDistributed
                from tensorflow.keras.layers import add, Layer
                from tensorflow.keras.layers import Lambda
                
                # Create a proper NotEqual layer
                class NotEqualLayer(Layer):
                    def __init__(self, **kwargs):
                        super(NotEqualLayer, self).__init__(**kwargs)
                    
                    def call(self, inputs):
                        x, y = inputs
                        return tf.not_equal(x, y)
                
                # Comprehensive custom objects dictionary
                custom_objects = {
                    'Dense': Dense,
                    'Dropout': Dropout,
                    'Embedding': Embedding,
                    'LSTM': LSTM,
                    'RepeatVector': RepeatVector,
                    'TimeDistributed': TimeDistributed,
                    'add': add,
                    'NotEqual': NotEqualLayer,
                    'NotEqualLayer': NotEqualLayer,
                    'Lambda': Lambda
                }
                
                with custom_object_scope(custom_objects):
                    self.model = tf.keras.models.load_model(model_path)
            else:
                raise e
                
        print(f"Model loaded from {model_path}")
        
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"Tokenizer loaded from {tokenizer_path}")
        
        return self.model

class CaptionGenerator:
    def __init__(self, model, tokenizer, max_length, image_features_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_features_dict = image_features_dict
        
        # Create reverse word index
        self.reverse_word_index = {v: k for k, v in self.tokenizer.word_index.items()}
        
    def generate_caption(self, image_name, beam_width=3):
        """Generate caption for an image using beam search"""
        if image_name not in self.image_features_dict:
            return "Image not found in features dictionary"
        
        image_features = self.image_features_dict[image_name]
        
        # Start with start token
        start_token = self.tokenizer.word_index.get('<start>', 1)
        end_token = self.tokenizer.word_index.get('<end>', 2)
        
        # Initialize beam search
        beams = [([start_token], 0.0)]  # (sequence, score)
        
        for _ in range(self.max_length - 1):
            new_beams = []
            
            for sequence, score in beams:
                if sequence[-1] == end_token:
                    new_beams.append((sequence, score))
                    continue
                
                # Prepare input
                padded_sequence = np.zeros((1, self.max_length))
                padded_sequence[0, :len(sequence)] = sequence
                
                # Predict next token
                predictions = self.model.predict([np.array([image_features]), padded_sequence], verbose=0)
                predictions = predictions[0, len(sequence) - 1, :]
                
                # Get top beam_width predictions
                top_indices = np.argsort(predictions)[-beam_width:]
                
                for idx in top_indices:
                    new_sequence = sequence + [idx]
                    new_score = score + np.log(predictions[idx] + 1e-10)
                    new_beams.append((new_sequence, new_score))
            
            # Keep top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Select best caption
        best_sequence = beams[0][0]
        
        # Convert to text
        caption = []
        for idx in best_sequence:
            if idx == end_token:
                break
            if idx in self.reverse_word_index and idx != start_token:
                caption.append(self.reverse_word_index[idx])
        
        return ' '.join(caption)
    
    def generate_caption_greedy(self, image_name):
        """Generate caption using greedy search (faster but less optimal)"""
        if image_name not in self.image_features_dict:
            return "Image not found in features dictionary"
        
        image_features = self.image_features_dict[image_name]
        
        # Start with start token
        input_sequence = [self.tokenizer.word_index.get('<start>', 1)]
        
        for _ in range(self.max_length - 1):
            # Prepare input
            padded_sequence = np.zeros((1, self.max_length))
            padded_sequence[0, :len(input_sequence)] = input_sequence
            
            # Predict next token
            predictions = self.model.predict([np.array([image_features]), padded_sequence], verbose=0)
            predictions = predictions[0, len(input_sequence) - 1, :]
            
            # Get most likely token
            next_token = np.argmax(predictions)
            
            # Add to sequence
            input_sequence.append(next_token)
            
            # Stop if end token
            if next_token == self.tokenizer.word_index.get('<end>', 2):
                break
        
        # Convert to text
        caption = []
        for idx in input_sequence[1:]:  # Skip start token
            if idx == self.tokenizer.word_index.get('<end>', 2):
                break
            if idx in self.reverse_word_index:
                caption.append(self.reverse_word_index[idx])
        
        return ' '.join(caption)

if __name__ == "__main__":
    # Example usage
    print("Image Captioning Model")
    print("This script provides the model architecture and training functionality.")
    print("Use train_model.py to train the model on your data.")