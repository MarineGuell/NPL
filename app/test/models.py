'''
Contient les implémentations des différents modèles de classification

--> contient les différents modèles de ML/DL utilisés

Définit plusieurs classes :
    SupervisedClassifier : pour les modèles de ML classiques (Logistic Regression, SVM, etc.)
    DeepLearningClassifier : pour les modèles basés sur BERT
    RNNTextClassifier : pour les modèles RNN (LSTM, GRU)
    KerasTextClassifier : pour les modèles TensorFlow/Keras
Gère :
    L'architecture des modèles
    L'entraînement
    L'évaluation
    La sauvegarde/chargement des modèles
'''

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Bidirectional, Attention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import re

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_output):
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(attention_weights * lstm_output, dim=1)

class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, use_attention=True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, 
                          hidden_dim, 
                          num_layers=n_layers, 
                          bidirectional=bidirectional, 
                          batch_first=True,
                          dropout=dropout if n_layers > 1 else 0)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(hidden_dim * 2 if bidirectional else hidden_dim)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.use_attention:
            output = self.attention(output)
        else:
            if self.rnn.bidirectional:
                output = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                output = hidden[-1,:,:]
        
        output = self.layer_norm(output)
        output = self.dropout(output)
        return self.fc(output)

class RNNTextClassifier:
    def __init__(self, vocab_size=25000, embedding_dim=100, hidden_dim=256, 
                 output_dim=2, n_layers=2, bidirectional=True, dropout=0.5,
                 use_attention=True, learning_rate=0.001, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.use_attention = use_attention
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        
    def build_vocab(self, texts):
        """Construit le vocabulaire à partir des textes"""
        from collections import Counter
        import re
        
        # Tokenization simple
        tokens = []
        for text in texts:
            tokens.extend(re.findall(r'\b\w+\b', text.lower()))
        
        # Comptage des tokens
        counter = Counter(tokens)
        
        # Création du vocabulaire
        self.vocab = {word: idx for idx, (word, _) in enumerate(counter.most_common(self.vocab_size-2))}
        self.vocab['<pad>'] = len(self.vocab)
        self.vocab['<unk>'] = len(self.vocab)
        
        # Création du tokenizer
        def tokenize(text):
            return [self.vocab.get(word, self.vocab['<unk>']) 
                   for word in re.findall(r'\b\w+\b', text.lower())]
        
        self.tokenizer = tokenize
        
    def prepare_data(self, texts, labels):
        """Prépare les données pour l'entraînement"""
        class RNNTextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                tokens = self.tokenizer(text)
                return {
                    'text': torch.tensor(tokens, dtype=torch.long),
                    'length': len(tokens),
                    'label': torch.tensor(label, dtype=torch.long)
                }
        
        return RNNTextDataset(texts, labels, self.tokenizer)
    
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              batch_size=32, epochs=5, learning_rate=None):
        """Entraîne le modèle RNN avec des améliorations"""
        if self.tokenizer is None:
            self.build_vocab(train_texts)
        
        if learning_rate is not None:
            self.learning_rate = learning_rate
            
        self.model = RNNClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_layers=self.n_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
            pad_idx=self.vocab['<pad>'],
            use_attention=self.use_attention
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        train_dataset = self.prepare_data(train_texts, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        if val_texts is not None and val_labels is not None:
            val_dataset = self.prepare_data(val_texts, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn
            )
        
        best_val_loss = float('inf')
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                text = batch['text'].to(self.device)
                lengths = batch['length'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(text, lengths)
                loss = self.criterion(predictions, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_texts, val_labels)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_model.pt')
                
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {avg_train_loss:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {avg_train_loss:.4f}')
    
    def evaluate(self, texts, labels):
        """Évalue le modèle"""
        self.model.eval()
        dataset = self.prepare_data(texts, labels)
        loader = DataLoader(dataset, batch_size=32, collate_fn=self.collate_fn)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                text = batch['text'].to(self.device)
                lengths = batch['length'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(text, lengths)
                _, predicted = torch.max(predictions, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def predict(self, texts):
        """Fait des prédictions"""
        self.model.eval()
        dataset = self.prepare_data(texts, [0] * len(texts))  # Labels factices
        loader = DataLoader(dataset, batch_size=32, collate_fn=self.collate_fn)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                text = batch['text'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                predictions = self.model(text, lengths)
                probs = torch.softmax(predictions, dim=1)
                _, preds = torch.max(probs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def collate_fn(self, batch):
        """Fonction pour gérer les séquences de longueurs variables"""
        text = [item['text'] for item in batch]
        lengths = [item['length'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Padding
        max_len = max(lengths)
        padded_text = []
        for t in text:
            if len(t) < max_len:
                t = torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.long)])
            padded_text.append(t)
        
        return {
            'text': torch.stack(padded_text),
            'length': torch.tensor(lengths),
            'label': torch.stack(labels)
        }
    
    def save_model(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'vocab': self.vocab,
            'model_params': {
                'vocab_size': len(self.vocab),
                'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'n_layers': self.n_layers,
                'bidirectional': self.bidirectional,
                'dropout': self.dropout,
                'use_attention': self.use_attention
            }
        }, path)
    
    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        checkpoint = torch.load(path)
        self.vocab = checkpoint['vocab']
        self.build_vocab([])  # Réinitialise le tokenizer
        
        params = checkpoint['model_params']
        self.model = RNNClassifier(
            vocab_size=params['vocab_size'],
            embedding_dim=params['embedding_dim'],
            hidden_dim=params['hidden_dim'],
            output_dim=params['output_dim'],
            n_layers=params['n_layers'],
            bidirectional=params['bidirectional'],
            dropout=params['dropout'],
            pad_idx=self.vocab['<pad>'],
            use_attention=params['use_attention']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])

class SupervisedClassifier:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.classes = None
        
    def create_model(self, n_classes):
        """Crée le modèle de classification approprié"""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, multi_class='multinomial')
        elif self.model_type == 'svm':
            self.model = SVC(probability=True)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100)
        elif self.model_type == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
        else:
            raise ValueError(f"Type de modèle non supporté : {self.model_type}")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Entraîne le modèle"""
        self.classes = np.unique(y_train)
        self.create_model(len(self.classes))
        
        # Normalisation des features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Entraînement du modèle
        self.model.fit(X_train_scaled, y_train)
        
        # Évaluation sur l'ensemble de validation si fourni
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred)
            return accuracy, report
        return None, None

    def predict(self, X):
        """Fait des prédictions avec le modèle"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Retourne les probabilités de prédiction"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def save_model(self, path):
        """Sauvegarde le modèle"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'classes': self.classes
        }, path)

    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        saved_data = joblib.load(path)
        self.model = saved_data['model']
        self.scaler = saved_data['scaler']
        self.classes = saved_data['classes']

class DeepLearningClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_data(self, texts, labels):
        """Prépare les données pour l'entraînement"""
        return TextDataset(texts, labels, self.tokenizer)

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              batch_size=16, epochs=3, learning_rate=2e-5):
        """Entraîne le modèle"""
        train_dataset = self.prepare_data(train_texts, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            if val_texts is not None and val_labels is not None:
                val_accuracy = self.evaluate(val_texts, val_labels)
                print(f"Validation Accuracy: {val_accuracy:.4f}")

    def evaluate(self, texts, labels):
        """Évalue le modèle"""
        self.model.eval()
        dataset = self.prepare_data(texts, labels)
        loader = DataLoader(dataset, batch_size=32)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return accuracy_score(all_labels, all_preds)

    def predict(self, texts):
        """Fait des prédictions"""
        self.model.eval()
        dataset = self.prepare_data(texts, [0] * len(texts))  # Labels factices
        loader = DataLoader(dataset, batch_size=32)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)

    def save_model(self, path):
        """Sauvegarde le modèle"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model.to(self.device)

class KerasTextClassifier:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=100):
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = Tokenizer(num_words=max_words)
        self.model = None
        self.word2vec_model = None
        
    def preprocess_text(self, texts):
        """Prétraite les textes pour l'entraînement"""
        # Tokenization
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        
        # Entraînement du modèle Word2Vec
        self.word2vec_model = Word2Vec(tokenized_texts, 
                                     vector_size=self.embedding_dim,
                                     window=5,
                                     min_count=1,
                                     workers=4)
        
        # Création de la matrice d'embedding
        word_index = self.tokenizer.word_index
        embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
        
        for word, i in word_index.items():
            if i < self.max_words:
                try:
                    embedding_vector = self.word2vec_model.wv[word]
                    embedding_matrix[i] = embedding_vector
                except KeyError:
                    continue
        
        return embedding_matrix
    
    def build_model(self, num_classes, embedding_matrix=None):
        """Construit le modèle Keras"""
        model = Sequential([
            Embedding(self.max_words, self.embedding_dim, 
                     weights=[embedding_matrix] if embedding_matrix is not None else None,
                     input_length=self.max_len,
                     trainable=embedding_matrix is None),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def prepare_data(self, texts, labels):
        """Prépare les données pour l'entraînement"""
        # Tokenization
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Padding
        X = pad_sequences(sequences, maxlen=self.max_len)
        y = np.array(labels)
        
        return X, y
    
    def train(self, texts, labels, validation_split=0.2, epochs=10, batch_size=32):
        """Entraîne le modèle"""
        # Prétraitement des textes
        embedding_matrix = self.preprocess_text(texts)
        
        # Préparation des données
        X, y = self.prepare_data(texts, labels)
        
        # Construction du modèle
        num_classes = len(np.unique(labels))
        self.build_model(num_classes, embedding_matrix)
        
        # Entraînement
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def predict(self, texts):
        """Fait des prédictions"""
        # Préparation des données
        sequences = self.tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # Prédictions
        predictions = self.model.predict(X)
        return np.argmax(predictions, axis=1), predictions
    
    def evaluate(self, texts, labels):
        """Évalue le modèle"""
        X, y = self.prepare_data(texts, labels)
        loss, accuracy = self.model.evaluate(X, y)
        return accuracy
    
    def save_model(self, path):
        """Sauvegarde le modèle"""
        # Sauvegarde du modèle Keras
        self.model.save(f"{path}_keras.h5")
        
        # Sauvegarde du tokenizer
        tokenizer_config = {
            'word_index': self.tokenizer.word_index,
            'num_words': self.tokenizer.num_words
        }
        joblib.dump(tokenizer_config, f"{path}_tokenizer.pkl")
        
        # Sauvegarde du modèle Word2Vec
        self.word2vec_model.save(f"{path}_word2vec.model")
    
    def load_model(self, path):
        """Charge un modèle sauvegardé"""
        # Chargement du modèle Keras
        self.model = tf.keras.models.load_model(f"{path}_keras.h5")
        
        # Chargement du tokenizer
        tokenizer_config = joblib.load(f"{path}_tokenizer.pkl")
        self.tokenizer.word_index = tokenizer_config['word_index']
        self.tokenizer.num_words = tokenizer_config['num_words']
        
        # Chargement du modèle Word2Vec
        self.word2vec_model = Word2Vec.load(f"{path}_word2vec.model") 