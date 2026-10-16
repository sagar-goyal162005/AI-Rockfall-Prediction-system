"""
Deep Learning models for 3D point cloud analysis and time series prediction in geological applications.
Implements PointNet-based architectures for feature extraction and LSTM for temporal modeling.
"""

# Core imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning features will be disabled.")
    # Create dummy classes
    class nn:
        class Module: pass
        class Sequential: pass
        class Conv1d: pass
        class BatchNorm1d: pass
        class Linear: pass
        class Dropout: pass
        class AdaptiveMaxPool1d: pass
        class Identity: pass
        class ReLU: pass
        class Flatten: pass
    class torch:
        class Tensor: pass
        @staticmethod
        def bmm(*args): return None
        @staticmethod
        def max(*args): return [None]

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D point cloud processing will be disabled.")
    class o3d:
        class geometry:
            class PointCloud: pass

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# TensorFlow imports with error handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False
    # Create dummy classes to avoid import errors
    class Sequential: pass
    class LSTM: pass
    class Dense: pass
    class Dropout: pass
    class BatchNormalization: pass
    class EarlyStopping: pass
    class ModelCheckpoint: pass

import joblib
import os

# Hugging Face datasets integration
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    load_dataset = None
    print("Hugging Face datasets not available. Install with: pip install datasets")

@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    num_points: int = 1024  # Number of points per sample
    num_classes: int = 4  # Risk levels: LOW, MEDIUM, HIGH, CRITICAL
    feature_dim: int = 1024  # Feature dimension
    dropout_rate: float = 0.3
    use_batch_norm: bool = True

@dataclass
class LSTMConfig:
    """Configuration for LSTM time series model"""
    sequence_length: int = 10  # Number of time steps to look back
    num_features: int = 6  # Number of input features per time step
    lstm_units: int = 50  # Number of LSTM units
    dense_units: int = 25  # Number of dense layer units
    dropout_rate: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    validation_split: float = 0.2

class PointNetFeatureExtractor(nn.Module):
    """PointNet feature extraction backbone"""
    
    def __init__(self, config: ModelConfig):
        super(PointNetFeatureExtractor, self).__init__()
        self.config = config
        
        # Input transform network (T-Net)
        self.input_transform = self._build_tnet(3, 3)
        
        # Feature transform network
        self.feature_transform = self._build_tnet(64, 64)
        
        # MLP layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, config.feature_dim, 1)
        
        # Batch normalization
        if config.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(64)
            self.bn4 = nn.BatchNorm1d(128)
            self.bn5 = nn.BatchNorm1d(config.feature_dim)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def _build_tnet(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build transformation network (T-Net)"""
        return nn.Sequential(
            nn.Conv1d(input_dim, 64, 1),
            nn.BatchNorm1d(64) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, output_dim * output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, 3, num_points]
            
        Returns:
            Tuple of (global_features, point_features)
        """
        batch_size, _, num_points = x.size()
        
        # Input transformation
        input_trans = self.input_transform(x)
        input_trans = input_trans.view(batch_size, 3, 3)
        x = torch.bmm(x.transpose(2, 1), input_trans).transpose(2, 1)
        
        # First set of convolutions
        x = F.relu(self.bn1(self.conv1(x)) if self.config.use_batch_norm else self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)) if self.config.use_batch_norm else self.conv2(x))
        
        # Feature transformation
        feature_trans = self.feature_transform(x)
        feature_trans = feature_trans.view(batch_size, 64, 64)
        x = torch.bmm(x.transpose(2, 1), feature_trans).transpose(2, 1)
        
        # Point features
        point_features = x.clone()
        
        # Continue with convolutions
        x = F.relu(self.bn3(self.conv3(x)) if self.config.use_batch_norm else self.conv3(x))
        x = F.relu(self.bn4(self.conv4(x)) if self.config.use_batch_norm else self.conv4(x))
        x = self.bn5(self.conv5(x)) if self.config.use_batch_norm else self.conv5(x)
        
        # Global feature extraction
        global_features = torch.max(x, 2)[0]
        
        return global_features, point_features

class RockfallRiskClassifier(nn.Module):
    """PointNet-based classifier for rockfall risk assessment"""
    
    def __init__(self, config: ModelConfig):
        super(RockfallRiskClassifier, self).__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.feature_dim, 512),
            nn.BatchNorm1d(512) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256) if config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            x: Input tensor [batch_size, 3, num_points]
            
        Returns:
            Classification logits [batch_size, num_classes]
        """
        global_features, _ = self.feature_extractor(x)
        return self.classifier(global_features)

class GeologicalFeatureDetector(nn.Module):
    """PointNet-based detector for geological features"""
    
    def __init__(self, config: ModelConfig, num_geological_features: int = 8):
        super(GeologicalFeatureDetector, self).__init__()
        self.config = config
        self.num_features = num_geological_features
        
        # Feature extractor
        self.feature_extractor = PointNetFeatureExtractor(config)
        
        # Feature detection heads
        self.feature_detectors = nn.ModuleDict({
            'slope_stability': self._build_detection_head(),
            'crack_detection': self._build_detection_head(),
            'weathering_degree': self._build_detection_head(),
            'joint_spacing': self._build_detection_head(),
            'surface_roughness': self._build_detection_head(),
            'discontinuity_orientation': self._build_detection_head(),
            'block_size': self._build_detection_head(),
            'overhang_detection': self._build_detection_head()
        })
        
    def _build_detection_head(self) -> nn.Module:
        """Build a detection head for specific geological features"""
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, 256),
            nn.BatchNorm1d(256) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128) if self.config.use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(128, 1),  # Single output for regression/binary classification
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for geological feature detection
        
        Args:
            x: Input tensor [batch_size, 3, num_points]
            
        Returns:
            Dictionary of feature predictions
        """
        global_features, _ = self.feature_extractor(x)
        
        outputs = {}
        for feature_name, detector in self.feature_detectors.items():
            outputs[feature_name] = detector(global_features)
            
        return outputs

class PointCloudDLModel:
    """Main interface for deep learning models on point clouds"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available - DL features disabled")
            self.enabled = False
            return
            
        if not OPEN3D_AVAILABLE:
            self.logger.warning("Open3D not available - point cloud processing disabled")
            self.enabled = False
            return
            
        self.enabled = True
        
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Initialize models
            self.risk_classifier = RockfallRiskClassifier(self.config).to(self.device)
            self.feature_detector = GeologicalFeatureDetector(self.config).to(self.device)
            
            # Training state
            self.risk_classifier_trained = False
            self.feature_detector_trained = False
        except Exception as e:
            self.logger.error(f"Failed to initialize PointCloudDLModel: {e}")
            self.enabled = False
        
    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> torch.Tensor:
        """
        Preprocess point cloud for deep learning model
        
        Args:
            pcd: Open3D point cloud
            
        Returns:
            Tensor ready for model input [1, 3, num_points]
        """
        if not self.enabled:
            self.logger.warning("PointCloudDLModel not enabled - returning dummy tensor")
            return torch.zeros(1, 3, 1024) if TORCH_AVAILABLE else None
            
        try:
            points = np.asarray(pcd.points)
            
            # Normalize to unit sphere
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
            
            # Sample or pad to fixed number of points
            if len(points) > self.config.num_points:
                # Random sampling
                indices = np.random.choice(len(points), self.config.num_points, replace=False)
                points = points[indices]
            elif len(points) < self.config.num_points:
                # Pad with duplicated points
                padding_needed = self.config.num_points - len(points)
                padding_indices = np.random.choice(len(points), padding_needed, replace=True)
                padding_points = points[padding_indices]
                points = np.vstack([points, padding_points])
            
            # Convert to tensor and transpose for conv1d
            tensor = torch.FloatTensor(points).transpose(0, 1).unsqueeze(0)  # [1, 3, num_points]
            return tensor.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to preprocess point cloud: {e}")
            return torch.zeros(1, 3, self.config.num_points) if TORCH_AVAILABLE else None
    
    def predict_risk(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """
        Predict rockfall risk from point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Dictionary with risk prediction and confidence
        """
        if not self.enabled:
            return {
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'probabilities': {'LOW': 0.25, 'MEDIUM': 0.25, 'HIGH': 0.25, 'CRITICAL': 0.25},
                'message': 'Deep learning model not available'
            }
            
        if not self.risk_classifier_trained:
            self.logger.warning("Risk classifier not trained, using dummy predictions")
            return {
                'risk_level': 'MEDIUM',
                'confidence': 0.5,
                'probabilities': {'LOW': 0.25, 'MEDIUM': 0.5, 'HIGH': 0.2, 'CRITICAL': 0.05}
            }
        
        try:
            self.risk_classifier.eval()
            with torch.no_grad():
                x = self.preprocess_point_cloud(pcd)
                if x is None:
                    raise Exception("Point cloud preprocessing failed")
                    
                logits = self.risk_classifier(x)
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
                risk_dict = {level: float(prob) for level, prob in zip(risk_levels, probabilities)}
                
                predicted_class = np.argmax(probabilities)
                predicted_risk = risk_levels[predicted_class]
                confidence = float(probabilities[predicted_class])
                
                return {
                    'risk_level': predicted_risk,
                    'confidence': confidence,
                    'probabilities': risk_dict
                }
        except Exception as e:
            self.logger.error(f"Risk prediction failed: {e}")
            return {
                'risk_level': 'UNKNOWN',
                'confidence': 0.0,
                'probabilities': {'LOW': 0.25, 'MEDIUM': 0.25, 'HIGH': 0.25, 'CRITICAL': 0.25},
                'error': str(e)
            }
    
    def detect_geological_features(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """
        Detect geological features from point cloud
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Dictionary of detected features and their scores
        """
        if not self.feature_detector_trained:
            self.logger.warning("Feature detector not trained, using dummy predictions")
            return {
                'slope_stability': 0.7,
                'crack_detection': 0.3,
                'weathering_degree': 0.4,
                'joint_spacing': 0.6,
                'surface_roughness': 0.5,
                'discontinuity_orientation': 0.8,
                'block_size': 0.6,
                'overhang_detection': 0.2
            }
        
        self.feature_detector.eval()
        with torch.no_grad():
            x = self.preprocess_point_cloud(pcd)
            features = self.feature_detector(x)
            
            # Convert to numpy and return
            result = {}
            for feature_name, tensor in features.items():
                result[feature_name] = float(tensor.cpu().numpy()[0, 0])
                
            return result
    
    def generate_synthetic_training_data(self, num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic training data for initial model training
        
        Args:
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of (data, labels)
        """
        data = []
        labels = []
        
        for _ in range(num_samples):
            # Generate random point cloud
            points = np.random.randn(self.config.num_points, 3)
            
            # Add some geological-like structure
            # Simulate slope
            slope_angle = np.random.uniform(0, 60)  # degrees
            points[:, 2] += points[:, 0] * np.tan(np.radians(slope_angle))
            
            # Add noise based on stability
            stability = np.random.uniform(0, 1)
            noise_level = (1 - stability) * 0.1
            points += np.random.normal(0, noise_level, points.shape)
            
            # Normalize
            centroid = np.mean(points, axis=0)
            points = points - centroid
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist
            
            # Assign risk level based on slope and stability
            if slope_angle > 45 and stability < 0.3:
                risk_level = 3  # CRITICAL
            elif slope_angle > 35 and stability < 0.5:
                risk_level = 2  # HIGH
            elif slope_angle > 20 and stability < 0.7:
                risk_level = 1  # MEDIUM
            else:
                risk_level = 0  # LOW
            
            data.append(points.transpose())  # [3, num_points]
            labels.append(risk_level)
        
        return torch.FloatTensor(data), torch.LongTensor(labels)
    
    def train_risk_classifier(self, num_epochs: int = 100, learning_rate: float = 0.001):
        """
        Train the risk classifier with synthetic data
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
        """
        self.logger.info("Training risk classifier with synthetic data")
        
        # Generate training data
        data, labels = self.generate_synthetic_training_data()
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(data, labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup training
        optimizer = torch.optim.Adam(self.risk_classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.risk_classifier.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_data, batch_labels in dataloader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.risk_classifier(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            if (epoch + 1) % 20 == 0:
                accuracy = 100 * correct / total
                self.logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader):.4f}, Accuracy: {accuracy:.2f}%')
        
        self.risk_classifier_trained = True
        self.logger.info("Risk classifier training completed")
    
    def save_models(self, save_dir: str):
        """Save trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.risk_classifier.state_dict(), 
                  os.path.join(save_dir, 'risk_classifier.pth'))


class RockfallLSTMPredictor:
    """LSTM-based time series predictor for rockfall risk assessment"""
    
    def __init__(self, config: LSTMConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
    def preprocess_csv_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess CSV data for LSTM training
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (sequences, labels)
        """
        # Read CSV and extract relevant rockfall data
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='latin-1')
        
        # Extract numerical features from the dataset
        # Look for columns that might contain temporal or geological data
        numerical_columns = []
        target_column = None
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                # Skip ID columns and very large numbers (likely file sizes)
                if 'id' not in col.lower() and 'size' not in col.lower():
                    if df[col].max() < 1000:  # Filter reasonable ranges
                        numerical_columns.append(col)
        
        # Create synthetic time series data if no clear temporal structure exists
        if len(numerical_columns) < self.config.num_features:
            # Generate synthetic rockfall-related features
            n_samples = min(1000, len(df))
            features = self._generate_synthetic_features(n_samples)
            target = self._generate_synthetic_target(n_samples)
        else:
            # Use available numerical data
            features = df[numerical_columns[:self.config.num_features]].values
            # Create target based on feature combinations or use last column
            target = self._create_risk_target(features)
        
        # Create sequences for LSTM
        sequences, labels = self._create_sequences(features, target)
        
        return sequences, labels
    
    def _generate_synthetic_features(self, n_samples: int) -> np.ndarray:
        """Generate synthetic rockfall monitoring features"""
        np.random.seed(42)
        
        # Simulate geological monitoring sensors
        displacement = np.random.normal(0, 0.5, n_samples)  # mm displacement
        strain = np.random.normal(100, 50, n_samples)  # microstrain
        pore_pressure = np.random.normal(60, 20, n_samples)  # kPa
        temperature = np.random.normal(15, 10, n_samples)  # celsius
        rainfall = np.random.exponential(2, n_samples)  # mm/hour
        seismic_activity = np.random.gamma(2, 0.5, n_samples)  # magnitude
        
        features = np.column_stack([
            displacement, strain, pore_pressure, 
            temperature, rainfall, seismic_activity
        ])
        
        return features
    
    def _generate_synthetic_target(self, n_samples: int) -> np.ndarray:
        """Generate synthetic risk labels"""
        # Create realistic risk distribution
        risk_levels = np.random.choice([0, 1, 2, 3], n_samples, 
                                     p=[0.5, 0.3, 0.15, 0.05])  # LOW, MEDIUM, HIGH, CRITICAL
        return risk_levels
    
    def _create_risk_target(self, features: np.ndarray) -> np.ndarray:
        """Create risk target from feature combinations"""
        # Simple risk calculation based on feature thresholds
        risk_score = np.mean(features, axis=1)
        risk_levels = np.zeros(len(risk_score))
        
        # Define risk thresholds
        low_threshold = np.percentile(risk_score, 25)
        medium_threshold = np.percentile(risk_score, 50)
        high_threshold = np.percentile(risk_score, 75)
        
        risk_levels[risk_score <= low_threshold] = 0  # LOW
        risk_levels[(risk_score > low_threshold) & (risk_score <= medium_threshold)] = 1  # MEDIUM
        risk_levels[(risk_score > medium_threshold) & (risk_score <= high_threshold)] = 2  # HIGH
        risk_levels[risk_score > high_threshold] = 3  # CRITICAL
        
        return risk_levels.astype(int)
    
    def load_huggingface_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess Hugging Face Rockfall_Simulator dataset
        
        Returns:
            Tuple of (sequences, labels)
        """
        if not DATASETS_AVAILABLE:
            self.logger.warning("Hugging Face datasets not available. Using synthetic data.")
            return self._generate_synthetic_features(1000), self._generate_synthetic_target(1000)
        
        try:
            self.logger.info("Loading Hugging Face dataset: zhaoyiww/Rockfall_Simulator")
            
            # Load the dataset
            ds = load_dataset("zhaoyiww/Rockfall_Simulator")
            
            # Get train split (or use available split)
            if 'train' in ds:
                dataset = ds['train']
            else:
                # Use the first available split
                dataset = ds[list(ds.keys())[0]]
            
            self.logger.info(f"Dataset loaded with {len(dataset)} samples")
            
            # Convert to pandas DataFrame for easier manipulation
            df = dataset.to_pandas()
            
            # Extract features and targets
            feature_columns = []
            target_column = None
            
            # Look for common feature names in rockfall datasets
            common_features = [
                'displacement', 'strain', 'pressure', 'temperature', 'velocity',
                'acceleration', 'distance', 'angle', 'height', 'width', 'depth',
                'volume', 'mass', 'energy', 'force', 'stress', 'deformation',
                'slope', 'roughness', 'moisture', 'weather', 'precipitation'
            ]
            
            # Find relevant columns
            for col in df.columns:
                col_lower = col.lower()
                
                # Check if it's a numeric column
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Check if it matches common feature patterns
                    if any(feature in col_lower for feature in common_features):
                        feature_columns.append(col)
                    # Check for target/risk/label columns
                    elif any(term in col_lower for term in ['risk', 'label', 'target', 'class', 'category']):
                        target_column = col
            
            # If no specific columns found, use first numeric columns
            if len(feature_columns) < self.config.num_features:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_columns = numeric_cols[:self.config.num_features]
            
            # Ensure we have enough features
            if len(feature_columns) < self.config.num_features:
                self.logger.warning(f"Dataset has only {len(feature_columns)} features, but {self.config.num_features} required. Padding with synthetic data.")
                # Generate additional synthetic features
                synthetic_features = self._generate_synthetic_features(len(df))
                synthetic_df = pd.DataFrame(synthetic_features, columns=[f'synthetic_{i}' for i in range(synthetic_features.shape[1])])
                df = pd.concat([df, synthetic_df], axis=1)
                feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:self.config.num_features]
            
            # Extract features
            features = df[feature_columns[:self.config.num_features]].values
            
            # Extract or create target
            if target_column and target_column in df.columns:
                targets = df[target_column].values
                # Convert to risk levels if needed
                if targets.dtype == 'object' or len(np.unique(targets)) > 4:
                    targets = self._create_risk_target(features)
            else:
                targets = self._create_risk_target(features)
            
            # Create sequences
            sequences, labels = self._create_sequences(features, targets)
            
            self.logger.info(f"Created {len(sequences)} sequences from Hugging Face dataset")
            return sequences, labels
            
        except Exception as e:
            self.logger.error(f"Failed to load Hugging Face dataset: {e}")
            self.logger.info("Falling back to synthetic data generation")
            # Fallback to synthetic data
            features = self._generate_synthetic_features(1000)
            targets = self._generate_synthetic_target(1000)
            return self._create_sequences(features, targets)
    
    def combine_datasets(self, csv_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine CSV data and Hugging Face dataset for training
        
        Args:
            csv_path: Optional path to CSV file
            
        Returns:
            Combined sequences and labels
        """
        all_sequences = []
        all_labels = []
        
        # Load Hugging Face dataset
        hf_sequences, hf_labels = self.load_huggingface_dataset()
        all_sequences.append(hf_sequences)
        all_labels.append(hf_labels)
        self.logger.info(f"Added {len(hf_sequences)} sequences from Hugging Face dataset")
        
        # Load CSV data if provided
        if csv_path and os.path.exists(csv_path):
            try:
                csv_sequences, csv_labels = self.preprocess_csv_data(csv_path)
                all_sequences.append(csv_sequences)
                all_labels.append(csv_labels)
                self.logger.info(f"Added {len(csv_sequences)} sequences from CSV data")
            except Exception as e:
                self.logger.warning(f"Failed to load CSV data: {e}")
        
        # Combine all datasets
        if len(all_sequences) > 0:
            combined_sequences = np.concatenate(all_sequences, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
        else:
            # Fallback to synthetic data
            features = self._generate_synthetic_features(1000)
            targets = self._generate_synthetic_target(1000)
            combined_sequences, combined_labels = self._create_sequences(features, targets)
        
        self.logger.info(f"Total combined dataset: {len(combined_sequences)} sequences")
        return combined_sequences, combined_labels
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequences = []
        labels = []
        
        for i in range(self.config.sequence_length, len(features)):
            sequences.append(features[i-self.config.sequence_length:i])
            labels.append(target[i])
        
        return np.array(sequences), np.array(labels)
    
    def build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.config.lstm_units, 
                 return_sequences=True,
                 input_shape=(self.config.sequence_length, self.config.num_features)),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            Dense(self.config.dense_units, activation='relu'),
            Dropout(self.config.dropout_rate),
            
            Dense(4, activation='softmax')  # 4 risk classes
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, csv_path: str = None, save_path: str = None) -> Dict:
        """
        Train the LSTM model using combined datasets
        
        Args:
            csv_path: Optional path to additional CSV training data
            save_path: Path to save trained model
            
        Returns:
            Training history
        """
        self.logger.info("Starting LSTM model training with combined datasets...")
        
        # Use combined datasets (Hugging Face + CSV)
        sequences, labels = self.combine_datasets(csv_path)
        
        # Normalize features
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = self.scaler.fit_transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences_scaled, labels, test_size=0.2, random_state=42
        )
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
        ]
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            callbacks.append(ModelCheckpoint(save_path, save_best_only=True))
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        
        # Save scaler
        if save_path:
            scaler_path = save_path.replace('.h5', '_scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        
        self.logger.info("LSTM model training completed")
        return history.history
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on a sequence
        
        Args:
            sequence: Input sequence [sequence_length, num_features]
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Normalize sequence
        sequence_scaled = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
        sequence_scaled = sequence_scaled.reshape(1, sequence.shape[0], sequence.shape[1])
        
        # Predict
        predictions = self.model.predict(sequence_scaled, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return predicted_class, confidence
    
    def predict_batch(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make batch predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Normalize sequences
        sequences_reshaped = sequences.reshape(-1, sequences.shape[-1])
        sequences_scaled = self.scaler.transform(sequences_reshaped)
        sequences_scaled = sequences_scaled.reshape(sequences.shape)
        
        # Predict
        predictions = self.model.predict(sequences_scaled, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        return predicted_classes, confidences
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load trained model"""
        self.model = tf.keras.models.load_model(model_path)
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        elif scaler_path is None:
            scaler_path = model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
        
        self.is_trained = True
        self.logger.info(f"LSTM model loaded from {model_path}")
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model:
            import io
            stream = io.StringIO()
            self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
            return stream.getvalue()
        return "Model not built yet"
    
    @staticmethod
    def risk_level_to_string(risk_level: int) -> str:
        """Convert risk level integer to string"""
        risk_map = {0: 'LOW', 1: 'MEDIUM', 2: 'HIGH', 3: 'CRITICAL'}
        return risk_map.get(risk_level, 'UNKNOWN')
    def save_models(self, save_dir: str):
        """Save trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save(self.risk_classifier.state_dict(), 
                  os.path.join(save_dir, 'risk_classifier.pth'))
        torch.save(self.feature_detector.state_dict(), 
                  os.path.join(save_dir, 'feature_detector.pth'))
        
        self.logger.info(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str):
        """Load trained models"""
        if not self.enabled:
            self.logger.warning("PointCloudDLModel not enabled - cannot load models")
            return False
            
        import os
        
        risk_path = os.path.join(save_dir, 'risk_classifier.pth')
        feature_path = os.path.join(save_dir, 'feature_detector.pth')
        
        try:
            if os.path.exists(risk_path):
                self.risk_classifier.load_state_dict(torch.load(risk_path, map_location=self.device))
                self.risk_classifier_trained = True
                self.logger.info("Risk classifier loaded")
            
            if os.path.exists(feature_path):
                self.feature_detector.load_state_dict(torch.load(feature_path, map_location=self.device))
                self.feature_detector_trained = True
                self.logger.info("Feature detector loaded")
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False


class RockfallLSTMPredictor:
    """LSTM-based time series predictor for rockfall risk assessment"""
    
    def __init__(self, config: LSTMConfig):
        if not TENSORFLOW_AVAILABLE:
            self.logger = None
            self.enabled = False
            return
            
        self.config = config
        self.enabled = True
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def preprocess_csv_data(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess CSV data for LSTM training"""
        if not self.enabled:
            self.logger.warning("LSTM predictor not enabled")
            return np.array([]), np.array([])
            
        try:
            df = pd.read_csv(csv_path)
            
            # Extract features and target
            feature_columns = ['displacement', 'strain', 'pore_pressure', 'temperature', 'rainfall', 'wind_speed']
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) < 3:
                self.logger.warning("Insufficient feature columns in CSV, using synthetic data")
                return self._generate_synthetic_features(1000), self._generate_synthetic_target(1000)
            
            features = df[available_columns].values
            
            # Create target variable if not present
            if 'risk_level' in df.columns:
                target = self.label_encoder.fit_transform(df['risk_level'])
            else:
                target = self._create_risk_target(features)
            
            return features, target
            
        except Exception as e:
            self.logger.error(f"Error preprocessing CSV: {e}")
            return self._generate_synthetic_features(1000), self._generate_synthetic_target(1000)
    
    def _generate_synthetic_features(self, n_samples: int) -> np.ndarray:
        """Generate synthetic feature data for training"""
        np.random.seed(42)
        features = np.random.normal(
            loc=[2.0, 100.0, 50.0, 15.0, 5.0, 10.0],  # displacement, strain, pressure, temp, rainfall, wind
            scale=[0.5, 20.0, 10.0, 5.0, 2.0, 3.0],
            size=(n_samples, 6)
        )
        
        # Add some realistic correlations
        for i in range(1, n_samples):
            features[i] = 0.8 * features[i-1] + 0.2 * features[i]  # temporal correlation
            
        return features
    
    def _generate_synthetic_target(self, n_samples: int) -> np.ndarray:
        """Generate synthetic target data"""
        np.random.seed(42)
        return np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])  # LOW, MEDIUM, HIGH, CRITICAL
    
    def _create_risk_target(self, features: np.ndarray) -> np.ndarray:
        """Create risk target based on feature values"""
        # Simple heuristic: higher displacement and strain = higher risk
        risk_score = (features[:, 0] / 5.0) + (features[:, 1] / 200.0)  # normalized displacement + strain
        
        # Convert to risk levels
        target = np.zeros(len(features), dtype=int)
        target[risk_score > 1.5] = 3  # CRITICAL
        target[(risk_score > 1.0) & (risk_score <= 1.5)] = 2  # HIGH
        target[(risk_score > 0.5) & (risk_score <= 1.0)] = 1  # MEDIUM
        # target <= 0.5 remains 0 (LOW)
        
        return target
    
    def load_huggingface_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load geological monitoring dataset from Hugging Face"""
        if not DATASETS_AVAILABLE:
            self.logger.warning("Hugging Face datasets not available, using synthetic data")
            return self._generate_synthetic_features(5000), self._generate_synthetic_target(5000)
        
        try:
            # Try to load a geological or environmental monitoring dataset
            # Note: This is a placeholder - replace with actual geological dataset
            dataset_names = [
                "environmental-monitoring/geological-sensors",
                "climate-data/environmental-monitoring", 
                "earth-observation/geological-stability"
            ]
            
            dataset = None
            for name in dataset_names:
                try:
                    dataset = load_dataset(name, split='train[:1000]')
                    break
                except:
                    continue
            
            if dataset is None:
                self.logger.info("No suitable HuggingFace dataset found, generating synthetic data")
                return self._generate_synthetic_features(5000), self._generate_synthetic_target(5000)
            
            # Convert dataset to numpy arrays
            features = []
            targets = []
            
            for sample in dataset:
                # Extract relevant features (this would depend on the actual dataset structure)
                feature_vector = [
                    sample.get('displacement', np.random.normal(2.0, 0.5)),
                    sample.get('strain', np.random.normal(100.0, 20.0)),
                    sample.get('pressure', np.random.normal(50.0, 10.0)),
                    sample.get('temperature', np.random.normal(15.0, 5.0)),
                    sample.get('rainfall', np.random.normal(5.0, 2.0)),
                    sample.get('wind_speed', np.random.normal(10.0, 3.0))
                ]
                
                # Create or extract target
                if 'risk_level' in sample:
                    target = sample['risk_level']
                else:
                    # Create synthetic target based on features
                    risk_score = feature_vector[0] / 5.0 + feature_vector[1] / 200.0
                    if risk_score > 1.5:
                        target = 3
                    elif risk_score > 1.0:
                        target = 2
                    elif risk_score > 0.5:
                        target = 1
                    else:
                        target = 0
                
                features.append(feature_vector)
                targets.append(target)
            
            return np.array(features), np.array(targets)
            
        except Exception as e:
            self.logger.error(f"Error loading HuggingFace dataset: {e}")
            return self._generate_synthetic_features(5000), self._generate_synthetic_target(5000)
    
    def combine_datasets(self, csv_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Combine CSV data and HuggingFace data for comprehensive training"""
        if not self.enabled:
            return np.array([]), np.array([])
            
        # Load HuggingFace data
        hf_features, hf_targets = self.load_huggingface_dataset()
        
        # Load CSV data if provided
        if csv_path and os.path.exists(csv_path):
            csv_features, csv_targets = self.preprocess_csv_data(csv_path)
            
            # Ensure compatible shapes
            min_features = min(csv_features.shape[1], hf_features.shape[1])
            csv_features = csv_features[:, :min_features]
            hf_features = hf_features[:, :min_features]
            
            # Combine datasets
            combined_features = np.vstack([csv_features, hf_features])
            combined_targets = np.hstack([csv_targets, hf_targets])
        else:
            combined_features = hf_features
            combined_targets = hf_targets
        
        self.logger.info(f"Combined dataset shape: {combined_features.shape}")
        return combined_features, combined_targets
    
    def _create_sequences(self, features: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_sequences, y_sequences = [], []
        
        for i in range(self.config.sequence_length, len(features)):
            X_sequences.append(features[i-self.config.sequence_length:i])
            y_sequences.append(target[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture"""
        if not self.enabled:
            return None
            
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, 
                 input_shape=(self.config.sequence_length, self.config.num_features)),
            Dropout(self.config.dropout_rate),
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            Dropout(self.config.dropout_rate),
            Dense(self.config.dense_units, activation='relu'),
            BatchNormalization(),
            Dense(4, activation='softmax')  # 4 risk levels
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, csv_path: str = None, save_path: str = None) -> Dict:
        """Train the LSTM model"""
        if not self.enabled:
            return {"error": "LSTM training not available - TensorFlow not installed"}
            
        try:
            # Prepare data
            features, targets = self.combine_datasets(csv_path)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Create sequences
            X_sequences, y_sequences = self._create_sequences(features_scaled, targets)
            
            # Split data
            split_idx = int(0.8 * len(X_sequences))
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
            
            # Build and train model
            self.model = self.build_model()
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
            ]
            
            if save_path:
                callbacks.append(ModelCheckpoint(save_path, save_best_only=True))
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config.epochs,
                batch_size=self.config.batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            results = {
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "training_samples": len(X_train),
                "validation_samples": len(X_test),
                "epochs_trained": len(history.history['loss']),
                "model_saved": save_path is not None
            }
            
            self.logger.info(f"Training completed: {results}")
            return results
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg}
    
    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """Make prediction on a single sequence"""
        if not self.enabled or self.model is None:
            return 0, 0.0
            
        try:
            # Ensure correct shape
            if sequence.ndim == 2:
                sequence = sequence.reshape(1, *sequence.shape)
            
            # Scale features
            scaled_sequence = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
            scaled_sequence = scaled_sequence.reshape(sequence.shape)
            
            # Predict
            prediction = self.model.predict(scaled_sequence, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            
            return int(predicted_class), confidence
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 0, 0.0
    
    def predict_batch(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on multiple sequences"""
        if not self.enabled or self.model is None:
            return np.zeros(len(sequences)), np.zeros(len(sequences))
            
        try:
            # Scale features
            scaled_sequences = self.scaler.transform(sequences.reshape(-1, sequences.shape[-1]))
            scaled_sequences = scaled_sequences.reshape(sequences.shape)
            
            # Predict
            predictions = self.model.predict(scaled_sequences, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            
            return predicted_classes, confidences
            
        except Exception as e:
            self.logger.error(f"Batch prediction failed: {e}")
            return np.zeros(len(sequences)), np.zeros(len(sequences))
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load pre-trained model and scaler"""
        if not self.enabled:
            return False
            
        try:
            self.model = tf.keras.models.load_model(model_path)
            
            if scaler_path and os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
            
            self.logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if not self.enabled or self.model is None:
            return "Model not available"
            
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    @staticmethod
    def create_default_config() -> LSTMConfig:
        """Create default LSTM configuration"""
        return LSTMConfig()
    
    def save_models(self, save_dir: str):
        """Save trained models and scaler"""
        if not self.enabled:
            return False
            
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            if self.model:
                model_path = os.path.join(save_dir, 'lstm_model.h5')
                self.model.save(model_path)
            
            scaler_path = os.path.join(save_dir, 'lstm_scaler.pkl')
            import joblib
            joblib.dump(self.scaler, scaler_path)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            return False
    
    def load_models(self, save_dir: str):
        """Load trained models and scaler"""
        if not self.enabled:
            return False
            
        try:
            model_path = os.path.join(save_dir, 'lstm_model.h5')
            scaler_path = os.path.join(save_dir, 'lstm_scaler.pkl')
            
            return self.load_model(model_path, scaler_path)
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
            return False