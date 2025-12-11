import numpy as np
import rasterio
from rasterio.plot import show
from skimage import filters, morphology
import cv2
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Polygon
import os
from tqdm import tqdm
from datetime import datetime
import math
import zipfile
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import psutil
from sklearn.model_selection import train_test_split

class CloudShadowDataset(Dataset):
    """Dataset for cloud and shadow detection."""
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transform:
            feature = self.transform(feature)
        
        if self.labels is not None:
            label = self.labels[idx]
            return torch.FloatTensor(feature), torch.LongTensor([label])
        else:
            return torch.FloatTensor(feature)

class CloudShadowDetector:
    def __init__(self, input_path, metadata=None, tile_size=1024):
        """
        Initialize the CloudShadowDetector with input satellite imagery.
        
        Args:
            input_path (str): Path to the input satellite image (GeoTIFF)
            metadata (dict): Optional metadata containing sun angles and other parameters
            tile_size (int): Size of tiles for processing large images
        """
        self.input_path = input_path
        self.tile_size = tile_size
        self.image = None
        self.metadata = metadata or {}
        self.cloud_mask = None
        self.shadow_mask = None
        self.toa_reflectance = None
        self.features = None
        self.model = None
        self.src = None  # Keep rasterio source open for tiled reading
        
        # Set device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        
    def load_image(self):
        """Load the satellite image metadata and prepare for tiled processing."""
        start_time = time.time()
        print("[INFO] Loading image metadata...")
        
        # Open the rasterio source for tiled reading
        self.src = rasterio.open(self.input_path)
        self.metadata.update(self.src.meta)
        
        end_time = time.time()
        print(f"[INFO] Image metadata loaded. Shape: {self.src.shape}")
        print(f"[INFO] Loading took {end_time - start_time:.2f} seconds")
        print(f"[INFO] Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
        return self
    
    def prepare_training_data(self, ground_truth_path=None):
        """Prepare training data from a subset of tiles to avoid memory issues."""
        print("[INFO] Preparing training data...")
        
        # Use only a subset of tiles for training to avoid memory issues
        height, width = self.src.shape
        n_tiles_h = (height + self.tile_size - 1) // self.tile_size
        n_tiles_w = (width + self.tile_size - 1) // self.tile_size
        
        # Use only first 100 tiles for training to avoid memory issues
        max_training_tiles = 100
        training_features = []
        training_labels = []
        
        print(f"[INFO] Extracting features from {max_training_tiles} tiles for training...")
        
        tile_count = 0
        with tqdm(total=max_training_tiles, desc="Processing training tiles") as pbar:
            for i in range(min(n_tiles_h, 10)):  # Limit to first 10 rows
                for j in range(min(n_tiles_w, 10)):  # Limit to first 10 columns
                    if tile_count >= max_training_tiles:
                        break
                    
                    # Calculate tile bounds
                    y_start = i * self.tile_size
                    y_end = min((i + 1) * self.tile_size, height)
                    x_start = j * self.tile_size
                    x_end = min((j + 1) * self.tile_size, width)
                    
                    # Read tile
                    tile_data = self.src.read(window=((y_start, y_end), (x_start, x_end)))
                    
                    # Process tile
                    tile_toa = self.dn_to_toa_tile(tile_data)
                    tile_features = self.extract_features_tile(tile_toa)
                    
                    # Use simple threshold-based labels for training
                    ndvi = tile_features[:, :, 0]
                    ndbi = tile_features[:, :, 1]
                    
                    # Create simple labels
                    cloud_mask = (ndvi > 0.3) | (ndbi > 0.1)
                    shadow_mask = (ndvi < -0.1) & (ndbi < -0.1)
                    
                    labels = np.zeros_like(ndvi, dtype=np.uint8)
                    labels[cloud_mask] = 1
                    labels[shadow_mask] = 2
                    
                    # Sample pixels for training (reduce memory usage)
                    sample_size = min(1000, tile_features.shape[0] * tile_features.shape[1])
                    indices = np.random.choice(
                        tile_features.shape[0] * tile_features.shape[1], 
                        sample_size, 
                        replace=False
                    )
                    
                    # Flatten and sample
                    flat_features = tile_features.reshape(-1, tile_features.shape[-1])
                    flat_labels = labels.flatten()
                    
                    training_features.append(flat_features[indices])
                    training_labels.append(flat_labels[indices])
                    
                    tile_count += 1
                    pbar.update(1)
                    
                    # Clear memory
                    del tile_data, tile_toa, tile_features, labels
                
                if tile_count >= max_training_tiles:
                    break
        
        # Combine all training data
        if training_features:
            features = np.vstack(training_features).astype(np.float32)
            labels = np.hstack(training_labels).astype(np.uint8)
            print(f"[INFO] Training data prepared: {features.shape[0]} samples")
        else:
            # Fallback: create minimal training data
            features = np.random.rand(100, 10).astype(np.float32)
            labels = np.random.randint(0, 3, 100).astype(np.uint8)
            print("[INFO] Using fallback training data")
        
        return features, labels
    
    def read_ground_truth_pixel(self, ground_truth_path, row, col):
        """Read a single pixel from ground truth image."""
        try:
            with rasterio.open(ground_truth_path) as src:
                return src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        except:
            return 0  # Default to NOCLOUD if reading fails
    
    def train_model(self, features, labels, epochs=50, batch_size=32, validation_split=0.2):
        """Train the PyTorch model with the prepared data."""
        print("[INFO] Training model...")
        
        # Split data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=validation_split, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = CloudShadowDataset(X_train, y_train)
        val_dataset = CloudShadowDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        in_channels = features.shape[1]
        num_classes = 3
        self.model = CloudShadowNet(in_channels, num_classes).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_true_labels = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.squeeze().to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    predictions = torch.argmax(outputs, dim=1)
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true_labels.extend(batch_labels.cpu().numpy())
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Calculate validation metrics
            val_accuracy = accuracy_score(val_true_labels, val_predictions)
            val_precision = precision_score(val_true_labels, val_predictions, average='weighted')
            val_recall = recall_score(val_true_labels, val_predictions, average='weighted')
            val_f1 = f1_score(val_true_labels, val_predictions, average='weighted')
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  Val Accuracy: {val_accuracy:.4f}, Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("  [INFO] Best model saved!")
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("[INFO] Training complete!")
        
        return train_losses, val_losses
    
    def evaluate_model(self, features, labels):
        """Evaluate the trained model and return comprehensive metrics."""
        print("[INFO] Evaluating model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        # Create evaluation dataset
        eval_dataset = CloudShadowDataset(features, labels)
        eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch_features, batch_labels in eval_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.squeeze().to(self.device)
                
                outputs = self.model(batch_features)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics
        metrics['Overall_Accuracy'] = accuracy_score(all_labels, all_predictions)
        metrics['Overall_Precision'] = precision_score(all_labels, all_predictions, average='weighted')
        metrics['Overall_Recall'] = recall_score(all_labels, all_predictions, average='weighted')
        metrics['Overall_F1'] = f1_score(all_labels, all_predictions, average='weighted')
        
        # Per-class metrics
        class_names = ['NOCLOUD', 'CLOUD', 'SHADOW']
        for i, class_name in enumerate(class_names):
            y_true = (np.array(all_labels) == i)
            y_pred = (np.array(all_predictions) == i)
            
            metrics[f'{class_name}_Precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics[f'{class_name}_Recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics[f'{class_name}_F1'] = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate IoU
            intersection = np.logical_and(y_true, y_pred).sum()
            union = np.logical_or(y_true, y_pred).sum()
            metrics[f'{class_name}_IoU'] = intersection / union if union > 0 else 0
        
        # Print results
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {metrics['Overall_Accuracy']:.4f}")
        print(f"Overall F1-Score: {metrics['Overall_F1']:.4f}")
        print(f"Overall Precision: {metrics['Overall_Precision']:.4f}")
        print(f"Overall Recall: {metrics['Overall_Recall']:.4f}")
        print("\nPer-Class Metrics:")
        for class_name in class_names:
            print(f"\n{class_name}:")
            print(f"  Precision: {metrics[f'{class_name}_Precision']:.4f}")
            print(f"  Recall: {metrics[f'{class_name}_Recall']:.4f}")
            print(f"  F1-Score: {metrics[f'{class_name}_F1']:.4f}")
            print(f"  IoU: {metrics[f'{class_name}_IoU']:.4f}")
        print("="*50)
        
        return metrics
    
    def process_tile(self, tile_data):
        """Process a single tile through the entire pipeline."""
        # Convert DN to TOA for this tile
        tile_toa = self.dn_to_toa_tile(tile_data)
        
        # Extract features for this tile
        tile_features = self.extract_features_tile(tile_toa)
        
        # Run model prediction
        tile_prediction = self.predict_tile(tile_features)
        
        return tile_prediction
    
    def dn_to_toa_tile(self, tile_data):
        """Convert Digital Numbers to TOA reflectance for a tile."""
        # Accept both lower and capitalized keys for sun angles
        if 'sun_elevation' in self.metadata and 'sun_azimuth' in self.metadata:
            sun_elevation = self.metadata['sun_elevation']
            sun_azimuth = self.metadata['sun_azimuth']
        elif 'SunElevationAtCenter' in self.metadata and 'SunAzimuthAtCenter' in self.metadata:
            sun_elevation = self.metadata['SunElevationAtCenter']
            sun_azimuth = self.metadata['SunAzimuthAtCenter']
        else:
            raise ValueError("Sun angles not found in metadata")
        
        sun_zenith = 90 - sun_elevation
        
        # Calculate solar irradiance (ESUN) for each band
        esun = {
            1: 1969.0,  # Blue
            2: 1840.0,  # Green
            3: 1551.0,  # Red
            4: 1044.0,  # NIR
            5: 225.7,   # SWIR1
            6: 82.07    # SWIR2
        }
        
        # Convert to TOA reflectance with float32 to save memory
        tile_toa = np.zeros_like(tile_data, dtype=np.float32)
        for band_idx in range(tile_data.shape[0]):
            band_num = band_idx + 1
            if band_num in esun:
                # Convert to float32 explicitly to avoid float64
                tile_data_float = tile_data[band_idx].astype(np.float32)
                scale_factor = np.float32(self.metadata.get('scale_factor', 1))
                esun_val = np.float32(esun[band_num])
                sun_zenith_rad = np.float32(math.radians(sun_zenith))
                
                tile_toa[band_idx] = (
                    (np.float32(math.pi) * tile_data_float * scale_factor) /
                    (esun_val * np.cos(sun_zenith_rad))
                )
        
        return tile_toa
    
    def extract_features_tile(self, tile_toa):
        """Extract features for a single tile."""
        n_features = 10
        tile_features = np.zeros((tile_toa.shape[1], tile_toa.shape[2], n_features))
        
        # Spectral Indices
        if tile_toa.shape[0] >= 4:  # Ensure we have enough bands
            ndvi = (tile_toa[3] - tile_toa[2]) / (tile_toa[3] + tile_toa[2] + 1e-6)
            tile_features[:, :, 0] = ndvi
            
            if tile_toa.shape[0] >= 5:
                ndbi = (tile_toa[4] - tile_toa[3]) / (tile_toa[4] + tile_toa[3] + 1e-6)
                tile_features[:, :, 1] = ndbi
                
                ndsi = (tile_toa[1] - tile_toa[4]) / (tile_toa[1] + tile_toa[4] + 1e-6)
                tile_features[:, :, 2] = ndsi
        
        # Morphological Features
        for band_idx in range(min(3, tile_toa.shape[0])):
            gradient = filters.sobel(tile_toa[band_idx])
            tile_features[:, :, 3 + band_idx] = gradient
            
            local_var = filters.gaussian(tile_toa[band_idx], sigma=2)
            tile_features[:, :, 6 + band_idx] = local_var
        
        return tile_features
    
    def predict_tile(self, tile_features):
        """Run model prediction on a single tile."""
        if self.model is None:
            # Fallback to threshold-based approach
            ndvi = tile_features[:, :, 0]
            ndbi = tile_features[:, :, 1]
            
            # Simple cloud detection based on spectral indices
            cloud_mask = (ndvi > 0.3) | (ndbi > 0.1)
            shadow_mask = (ndvi < -0.1) & (ndbi < -0.1)
            
            # Create combined mask (0: NOCLOUD, 1: CLOUD, 2: SHADOW)
            combined_mask = np.zeros_like(ndvi, dtype=np.uint8)
            combined_mask[cloud_mask] = 1
            combined_mask[shadow_mask] = 2
            
            return combined_mask
        else:
            # Use trained PyTorch model for prediction
            self.model.eval()
            with torch.no_grad():
                # Process tile in batches if it's large
                tile_shape = tile_features.shape
                predictions = np.zeros((tile_shape[0], tile_shape[1]), dtype=np.uint8)
                
                # Process in smaller chunks to avoid memory issues
                chunk_size = 64
                for i in range(0, tile_shape[0], chunk_size):
                    for j in range(0, tile_shape[1], chunk_size):
                        end_i = min(i + chunk_size, tile_shape[0])
                        end_j = min(j + chunk_size, tile_shape[1])
                        
                        chunk_features = tile_features[i:end_i, j:end_j, :]
                        chunk_tensor = torch.from_numpy(chunk_features).float().to(self.device)
                        
                        # Add batch dimension if needed
                        if len(chunk_tensor.shape) == 3:
                            chunk_tensor = chunk_tensor.unsqueeze(0)
                        
                        outputs = self.model(chunk_tensor)
                        chunk_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                        
                        predictions[i:end_i, j:end_j] = chunk_predictions
                
                return predictions
    
    def process_image_tiled(self):
        """Process the entire image using tiles."""
        print("[INFO] Starting tiled processing...")
        start_time = time.time()
        
        # Calculate number of tiles
        height, width = self.src.shape
        n_tiles_h = (height + self.tile_size - 1) // self.tile_size
        n_tiles_w = (width + self.tile_size - 1) // self.tile_size
        total_tiles = n_tiles_h * n_tiles_w
        
        print(f"[INFO] Processing {total_tiles} tiles ({n_tiles_h}x{n_tiles_w})")
        
        # Initialize output arrays
        self.combined_mask = np.zeros((height, width), dtype=np.uint8)
        self.cloud_mask = np.zeros((height, width), dtype=bool)
        self.shadow_mask = np.zeros((height, width), dtype=bool)
        
        # Process tiles
        with tqdm(total=total_tiles, desc="Processing tiles") as pbar:
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    # Calculate tile bounds
                    y_start = i * self.tile_size
                    y_end = min((i + 1) * self.tile_size, height)
                    x_start = j * self.tile_size
                    x_end = min((j + 1) * self.tile_size, width)
                    
                    # Read tile
                    tile_data = self.src.read(window=((y_start, y_end), (x_start, x_end)))
                    
                    # Process tile
                    tile_result = self.process_tile(tile_data)
                    
                    # Store results
                    self.combined_mask[y_start:y_end, x_start:x_end] = tile_result
                    self.cloud_mask[y_start:y_end, x_start:x_end] = (tile_result == 1)
                    self.shadow_mask[y_start:y_end, x_start:x_end] = (tile_result == 2)
                    
                    pbar.update(1)
                    
                    # Print memory usage every 10 tiles
                    if (i * n_tiles_w + j) % 10 == 0:
                        mem_usage = psutil.Process().memory_info().rss / 1024 / 1024
                        print(f"[INFO] Memory usage: {mem_usage:.1f} MB")
        
        end_time = time.time()
        print(f"[INFO] Tiled processing complete in {end_time - start_time:.2f} seconds")
        return self
    
    def save_results(self, output_dir):
        """Save the detection results as georeferenced GeoTIFF and shapefiles."""
        print(f"[INFO] Saving results to {output_dir} ...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined mask as georeferenced GeoTIFF (8-bit)
        mask_path = os.path.join(output_dir, 'cloud_shadow_mask.tiff')
        
        # Copy metadata from source image
        metadata = self.metadata.copy()
        metadata.update({
            'count': 1,
            'dtype': 'uint8',
            'nodata': None
        })
        
        with rasterio.open(mask_path, 'w', **metadata) as dst:
            dst.write(self.combined_mask.astype('uint8'), 1)
        
        print(f"[INFO] Georeferenced mask saved: {mask_path}")
        print("[INFO] Class values: NOCLOUD=0, CLOUD=1, SHADOW=2")
        
        # Save shapefiles
        def mask_to_shapefile(mask, output_path):
            contours, _ = cv2.findContours(
                mask.astype('uint8'),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            polygons = []
            for contour in contours:
                if len(contour) > 2:
                    polygon = Polygon(contour.reshape(-1, 2))
                    if polygon.area > 100:
                        polygons.append(polygon)
            
            gdf = gpd.GeoDataFrame(geometry=polygons, crs=self.metadata['crs'])
            gdf.to_file(output_path)
        
        # Save cloud and shadow shapefiles
        cloud_shp = os.path.join(output_dir, 'cloud.shp')
        shadow_shp = os.path.join(output_dir, 'shadow.shp')
        
        mask_to_shapefile(self.cloud_mask, cloud_shp)
        mask_to_shapefile(self.shadow_mask, shadow_shp)
        
        # Create zip files
        with zipfile.ZipFile(os.path.join(output_dir, 'cloudshapes.zip'), 'w') as zipf:
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                zipf.write(cloud_shp.replace('.shp', ext), os.path.basename(cloud_shp.replace('.shp', ext)))
        
        with zipfile.ZipFile(os.path.join(output_dir, 'shadowshapes.zip'), 'w') as zipf:
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                zipf.write(shadow_shp.replace('.shp', ext), os.path.basename(shadow_shp.replace('.shp', ext)))
        
        print("[INFO] Results saved.")
        return self
    
    def __del__(self):
        """Clean up rasterio source."""
        if hasattr(self, 'src') and self.src is not None:
            self.src.close()

class CloudShadowNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CloudShadowNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.softmax(x, dim=1)
        return x

def process_image(input_path, output_dir, metadata=None, ground_truth_path=None, tile_size=1024, train_model=False):
    """
    Process a single satellite image to detect clouds and shadows using tiled processing.
    
    Args:
        input_path (str): Path to the input satellite image
        output_dir (str): Directory to save the output masks
        metadata (dict): Optional metadata containing sun angles and other parameters
        ground_truth_path (str): Optional path to ground truth mask for training
        tile_size (int): Size of tiles for processing large images
        train_model (bool): Whether to train the model or use threshold-based detection
    """
    detector = CloudShadowDetector(input_path, metadata, tile_size)
    detector.load_image()
    
    if train_model:
        print("[INFO] Training mode enabled - preparing training data...")
        # Prepare training data
        features, labels = detector.prepare_training_data(ground_truth_path)
        
        # Train model
        train_losses, val_losses = detector.train_model(features, labels)
        
        # Evaluate model
        metrics = detector.evaluate_model(features, labels)
        
        # Save training results
        results_df = pd.DataFrame([metrics])
        results_df.to_csv(os.path.join(output_dir, 'evaluation_metrics.csv'), index=False)
        print(f"[INFO] Evaluation metrics saved to {output_dir}/evaluation_metrics.csv")
    else:
        print("[INFO] Using threshold-based detection (no training required)")
    
    # Process the full image
    detector.process_image_tiled()
    detector.save_results(output_dir)
    
    return detector 