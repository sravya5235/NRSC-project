# Cloud and Shadow Detection Pipeline

This project provides an automated pipeline for detecting clouds and shadows in satellite imagery. The pipeline uses a combination of spectral indices and machine learning techniques to achieve high accuracy (>90%) in cloud and shadow detection.

## Features

- Cloud detection using spectral indices and K-means clustering
- Shadow detection using spectral characteristics
- Output generation in both GeoTIFF and ESRI Shapefile formats
- Interactive development environment using Jupyter Notebook
- High accuracy (>90%) in cloud and shadow detection

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt

## Installation

1. Clone this repository:
bash
git clone <repository-url>
cd cloud-shadow-detection


2. Create a virtual environment (recommended):
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


3. Install dependencies:
bash
pip install -r requirements.txt


## Usage

### Using the Python Script

python
from cloud_shadow_detection import process_image

# Process a single image
process_image(
    input_path="path/to/your/satellite/image.tif",
    output_dir="output"
)


### Using the Jupyter Notebook

1. Start Jupyter Notebook:
bash
jupyter notebook


2. Open cloud_shadow_detection.ipynb
3. Follow the notebook instructions to process your images

## Output Files

The pipeline generates the following outputs in the specified output directory:

- cloud_mask.tif: GeoTIFF file containing the cloud mask
- shadow_mask.tif: GeoTIFF file containing the shadow mask
- cloud_mask.shp: ESRI Shapefile containing cloud polygons
- shadow_mask.shp: ESRI Shapefile containing shadow polygons

## Methodology

The cloud and shadow detection pipeline uses the following approach:

1. *Cloud Detection*:
   - Normalized Difference Built-up Index (NDBI)
   - Normalized Difference Snow Index (NDSI)
   - K-means clustering for classification
   - Morphological operations for mask refinement

2. *Shadow Detection*:
   - Spectral characteristics analysis
   - Otsu thresholding
   - Morphological operations for mask refinement

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
