"""Feature extraction utilities for trained models."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from recbole.quick_start import load_data_and_model


class FNNFeatureExtractor:
    """Extract user x item features from trained FNN model"""
    
    def __init__(self, model_path: str, dataset_name: str = "click_prediction"):
        """
        Initialize feature extractor
        
        Args:
            model_path: Path to saved model file
            dataset_name: Name of dataset used for training
        """
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.config = None
        self.model = None
        self.dataset = None
        self.train_data = None
        self.valid_data = None
        self.test_data = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load trained model and dataset"""
        try:
            print(f"Loading model from: {self.model_path}")
            
            # Load model and configuration
            self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = load_data_and_model(
                model_file=self.model_path
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully: {self.model.__class__.__name__}")
            print(f"Device: {self.config['device']}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def extract_user_item_features(
        self, 
        data_type: str = "test",
        batch_size: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract user x item feature vectors from FNN model
        
        Args:
            data_type: Type of data to extract features from ("train", "valid", "test")
            batch_size: Batch size for processing (None uses default)
            save_path: Path to save extracted features (optional)
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        # Select data source
        if data_type == "train":
            data_loader = self.train_data
        elif data_type == "valid":
            data_loader = self.valid_data
        elif data_type == "test":
            data_loader = self.test_data
        else:
            raise ValueError(f"Invalid data_type: {data_type}. Use 'train', 'valid', or 'test'")
        
        if batch_size:
            data_loader.batch_size = batch_size
        
        print(f"Extracting features from {data_type} data...")
        
        # Storage for extracted features
        all_user_ids = []
        all_item_ids = []
        all_embeddings = []
        all_mlp_features = []
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, interaction in enumerate(data_loader):
                # Move to device
                interaction = interaction.to(self.config['device'])
                
                # Extract user and item IDs
                user_ids = interaction[self.dataset.uid_field].cpu().numpy()
                item_ids = interaction[self.dataset.iid_field].cpu().numpy()
                
                # Extract labels if available
                if self.dataset.label_field in interaction:
                    labels = interaction[self.dataset.label_field].cpu().numpy()
                else:
                    labels = np.zeros(len(user_ids))
                
                # Extract concatenated embeddings (main user-item features)
                embeddings = self.model.concat_embed_input_fields(interaction)
                
                # Extract MLP features (intermediate representations)
                flattened_input = torch.flatten(embeddings, start_dim=1)
                mlp_features = self.model.mlp_layers[:-1](flattened_input)  # Exclude final layer
                
                # Get predictions
                predictions = self.model.predict(interaction).cpu().numpy()
                
                # Store results
                all_user_ids.extend(user_ids)
                all_item_ids.extend(item_ids)
                all_embeddings.append(embeddings.cpu().numpy())
                all_mlp_features.append(mlp_features.cpu().numpy())
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {batch_idx + 1} batches...")
        
        # Concatenate all features
        embeddings_array = np.vstack(all_embeddings)
        mlp_features_array = np.vstack(all_mlp_features)
        
        # Create result dictionary
        features = {
            'user_ids': np.array(all_user_ids),
            'item_ids': np.array(all_item_ids),
            'embeddings': embeddings_array,  # Raw concatenated embeddings
            'mlp_features': mlp_features_array,  # MLP intermediate features
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels),
            'embedding_dim': embeddings_array.shape[1],
            'mlp_feature_dim': mlp_features_array.shape[1],
            'num_samples': len(all_user_ids)
        }
        
        print("Feature extraction completed:")
        print(f"  - Samples: {features['num_samples']}")
        print(f"  - Embedding dimension: {features['embedding_dim']}")
        print(f"  - MLP feature dimension: {features['mlp_feature_dim']}")
        
        # Save features if path provided
        if save_path:
            self._save_features(features, save_path)
        
        return features
    
    def extract_user_item_embeddings(
        self, 
        user_ids: Union[List[int], np.ndarray], 
        item_ids: Union[List[int], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract embeddings for specific user-item pairs
        
        Args:
            user_ids: List or array of user IDs
            item_ids: List or array of item IDs
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        if len(user_ids) != len(item_ids):
            raise ValueError("user_ids and item_ids must have the same length")
        
        # Convert to tensors
        user_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.config['device'])
        item_tensor = torch.tensor(item_ids, dtype=torch.long, device=self.config['device'])
        
        with torch.no_grad():
            # Extract user embeddings
            if hasattr(self.model, 'user_embedding'):
                user_embeddings = self.model.user_embedding(user_tensor).cpu().numpy()
            else:
                # FNN doesn't have separate user embedding, extract from token embeddings
                user_embeddings = self.model.token_embedding_table[self.dataset.uid_field](user_tensor).cpu().numpy()
            
            # Extract item embeddings
            if hasattr(self.model, 'item_embedding'):
                item_embeddings = self.model.item_embedding(item_tensor).cpu().numpy()
            else:
                # FNN doesn't have separate item embedding, extract from token embeddings
                item_embeddings = self.model.token_embedding_table[self.dataset.iid_field](item_tensor).cpu().numpy()
        
        return user_embeddings, item_embeddings
    
    def analyze_feature_similarity(
        self, 
        features: Dict[str, np.ndarray], 
        top_k: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Analyze user-item feature similarity
        
        Args:
            features: Features extracted from extract_user_item_features
            top_k: Number of top similar pairs to return
            
        Returns:
            Dictionary with similarity analysis results
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        embeddings = features['embeddings']
        user_ids = features['user_ids']
        item_ids = features['item_ids']
        
        # Compute cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find top-k most similar pairs
        similarity_scores = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity_scores.append({
                    'user1': user_ids[i],
                    'item1': item_ids[i],
                    'user2': user_ids[j],
                    'item2': item_ids[j],
                    'similarity': similarity_matrix[i, j],
                    'idx1': i,
                    'idx2': j
                })
        
        # Sort by similarity and get top-k
        similarity_scores.sort(key=lambda x: x['similarity'], reverse=True)
        top_similar = similarity_scores[:top_k]
        
        return {
            'similarity_matrix': similarity_matrix,
            'top_similar_pairs': top_similar,
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix)
        }
    
    def _save_features(self, features: Dict[str, np.ndarray], save_path: str) -> None:
        """Save extracted features to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.npz':
            # Save as numpy compressed file
            np.savez_compressed(save_path, **features)
            print(f"Features saved to: {save_path}")
        
        elif save_path.suffix == '.csv':
            # Save as CSV file
            df = pd.DataFrame({
                'user_id': features['user_ids'],
                'item_id': features['item_ids'],
                'prediction': features['predictions'],
                'label': features['labels']
            })
            
            # Add embedding features as columns
            for i in range(features['embedding_dim']):
                df[f'embed_{i}'] = features['embeddings'][:, i]
            
            # Add MLP features as columns
            for i in range(features['mlp_feature_dim']):
                df[f'mlp_{i}'] = features['mlp_features'][:, i]
            
            df.to_csv(save_path, index=False)
            print(f"Features saved to: {save_path}")
        
        else:
            raise ValueError(f"Unsupported file format: {save_path.suffix}. Use .npz or .csv")
    
    def get_model_info(self) -> Dict[str, any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {}
        
        return {
            'model_type': self.model.__class__.__name__,
            'device': str(self.config['device']),
            'embedding_size': self.config.get('embedding_size', 'Unknown'),
            'mlp_hidden_size': self.config.get('mlp_hidden_size', 'Unknown'),
            'dataset_name': self.dataset_name,
            'num_users': self.dataset.user_num,
            'num_items': self.dataset.item_num,
            'num_interactions': len(self.dataset),
        }


def extract_fnn_features_cli(
    model_path: str,
    data_type: str = "test",
    save_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    analyze_similarity: bool = False
) -> Dict[str, np.ndarray]:
    """
    CLI function to extract FNN features
    
    Args:
        model_path: Path to saved FNN model
        data_type: Data split to extract features from
        save_path: Path to save features
        batch_size: Batch size for processing
        analyze_similarity: Whether to perform similarity analysis
        
    Returns:
        Extracted features dictionary
    """
    # Initialize extractor
    extractor = FNNFeatureExtractor(model_path)
    
    # Print model info
    model_info = extractor.get_model_info()
    print("\n" + "="*50)
    print("Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print("="*50 + "\n")
    
    # Extract features
    features = extractor.extract_user_item_features(
        data_type=data_type,
        batch_size=batch_size,
        save_path=save_path
    )
    
    # Perform similarity analysis if requested
    if analyze_similarity:
        print("\nPerforming similarity analysis...")
        similarity_results = extractor.analyze_feature_similarity(features, top_k=5)
        
        print(f"Mean similarity: {similarity_results['mean_similarity']:.4f}")
        print(f"Std similarity: {similarity_results['std_similarity']:.4f}")
        print("\nTop 5 most similar user-item pairs:")
        for i, pair in enumerate(similarity_results['top_similar_pairs']):
            print(f"  {i+1}. User {pair['user1']}-Item {pair['item1']} ~ "
                  f"User {pair['user2']}-Item {pair['item2']} "
                  f"(similarity: {pair['similarity']:.4f})")
    
    return features