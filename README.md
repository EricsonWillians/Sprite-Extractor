# Sprite Extractor

A sophisticated sprite extraction tool leveraging advanced computer vision techniques for robust sprite sheet analysis and extraction. Built with OpenCV and Python, it implements multiple extraction strategies with configurable parameters for optimal results.

## Technical Overview

### Core Technologies
- **OpenCV**: Primary computer vision framework for image processing and contour detection
- **NumPy**: Efficient array operations and mathematical computations
- **Rich**: Advanced terminal output and progress tracking
- **Poetry**: Dependency and package management

### Architecture Components

#### 1. Image Processing Pipeline
- **Preprocessing**
  - Bilateral filtering for noise reduction while preserving edges
  - Alpha channel handling with RGB correlation
  - Adaptive thresholding using Otsu's method
  - Morphological operations for edge enhancement

#### 2. Sprite Detection Methods

##### Contour-based Detection
- **Algorithm**: Hierarchical contour detection using `cv2.RETR_TREE`
- **Approximation**: `cv2.CHAIN_APPROX_TC89_KCOS` for precise contour representation
- **Filtering Criteria**:
  ```python
  - Area ratio: sprite_area / bounding_box_area > 0.15
  - Complexity: 4 <= vertices <= 100
  - Solidity: convex_hull_area / contour_area > 0.2
  - Aspect ratio: 0.1 <= width/height <= 10
  ```

##### Grid-based Detection
- Uniform grid partitioning with configurable cell dimensions
- Non-zero pixel analysis for sprite presence
- Transparency awareness for RGBA images

#### 3. Duplicate Detection System

##### Sprite Signature Computation
- **Perceptual Hash (pHash)**
  - DCT-based hash generation
  - 64x64 grayscale normalization
  - 8x8 frequency coefficient matrix
  
- **Feature Vectors**
  ```python
  {
      'phash': binary_hash,  # 64-bit perceptual hash
      'histogram': hist,     # 16-bin intensity histogram
      'edges': edge_sum,     # Canny edge detection signature
      'moments': hu_moments, # 7 Hu moment invariants
      'pattern': binary_pat  # Non-zero pixel pattern
  }
  ```

##### Similarity Analysis
- **Weighted Multi-criteria Comparison**
  ```python
  weights = {
      'phash': 0.4,       # Perceptual hash comparison
      'histogram': 0.2,   # Histogram correlation
      'edges': 0.1,       # Edge pattern similarity
      'moments': 0.2,     # Shape moment analysis
      'dimensions': 0.1   # Size ratio comparison
  }
  ```

##### Quality Assessment
- **Sprite Selection Criteria**
  ```python
  score = (
      0.4 * normalized_area +
      0.3 * edge_density +
      0.3 * position_score
  )
  ```

## Installation

### Poetry Installation (Recommended)
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Clone and install project
git clone https://github.com/ericsonwillians/sprite-extractor.git
cd sprite-extractor
poetry install
```

### Development Environment Setup
```bash
# Create virtual environment
poetry shell

# Install development dependencies
poetry install --with dev
```

## Technical Usage Guide

### Command-line Interface

#### Basic Extraction
```bash
poetry run sprite-extractor <input_path> <output_path> [options]
```

#### Advanced Configuration
```bash
poetry run sprite-extractor input.png output/ \
    --threshold 2 \         # Binary threshold value
    --min-size 4 \         # Minimum sprite dimension
    --max-size 150 \       # Maximum sprite dimension
    --padding 3 \          # Padding pixels
    --format png \         # Output format
    --metadata \           # Generate metadata
    --debug               # Enable debug logging
```

### Parameter Optimization

#### Threshold Selection
- **Low values** (1-5): Best for clean sprite sheets with sharp edges
- **Medium values** (5-20): Suitable for anti-aliased sprites
- **High values** (20-50): Required for noisy or textured sprites

#### Size Constraints
```python
min_size = min(sprite_dimensions) * 0.5  # Minimum dimension
max_size = max(sprite_dimensions) * 1.5  # Maximum dimension
```

#### Edge Cases
1. **Transparent Sprites**
   ```python
   # Alpha channel processing
   alpha = image[:, :, 3]
   rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
   combined = cv2.addWeighted(alpha, 0.7, rgb, 0.3, 0)
   ```

2. **Overlapping Sprites**
   ```python
   # Overlap detection
   intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * \
                 max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
   overlap_ratio = intersection / min(w1 * h1, w2 * h2)
   ```

### Output Specifications

#### Image Formats
- **PNG**: Lossless compression with alpha channel
  ```python
  cv2.imwrite(filename, sprite, [cv2.IMWRITE_PNG_COMPRESSION, 9])
  ```
- **WebP**: Lossy compression with quality control
  ```python
  cv2.imwrite(filename, sprite, [cv2.IMWRITE_WEBP_QUALITY, 95])
  ```

#### Metadata Schema
```json
{
  "sprites": [{
    "index": int,
    "x": int,
    "y": int,
    "width": int,
    "height": int,
    "area": int,
    "signature": {
      "phash": bytes,
      "histogram": float[16],
      "edges": float,
      "moments": float[7]
    }
  }],
  "extraction_mode": "contour" | "grid",
  "settings": {
    "threshold": int,
    "padding": int,
    "min_size": int,
    "max_size": int | null,
    "format": "png" | "jpg" | "webp",
    "auto_crop": boolean
  }
}
```

## Performance Considerations

### Time Complexity
- Contour detection: O(n) where n = image pixels
- Duplicate detection: O(k²) where k = number of sprites
- Signature computation: O(w×h) per sprite

### Memory Usage
- Base image: width × height × channels bytes
- Contour storage: ~8 bytes per contour point
- Signature storage: ~128 bytes per sprite

### Optimization Strategies
1. **Preprocessing**
   - Downscale large images before processing
   - Use bilateral filtering selectively
   
2. **Contour Detection**
   - Adjust approximation method based on sprite complexity
   - Implement early filtering of invalid contours

3. **Duplicate Detection**
   - Implement spatial partitioning for large sprite sheets
   - Use approximate nearest neighbor search for similarity

## Error Handling and Logging

### Log Levels
```python
logging.DEBUG    # Detailed processing information
logging.INFO     # General progress updates
logging.WARNING  # Non-critical issues
logging.ERROR    # Critical failures
```

### Exception Hierarchy
```python
class SpriteExtractorError(Exception): pass
class ImageLoadError(SpriteExtractorError): pass
class ContourDetectionError(SpriteExtractorError): pass
class DuplicateDetectionError(SpriteExtractorError): pass
```

## Testing

### Unit Tests
```bash
poetry run pytest tests/
```

### Coverage Analysis
```bash
poetry run pytest --cov=src tests/
```

## Contributing

### Development Workflow
1. Fork repository
2. Create feature branch
3. Implement changes with tests
4. Run full test suite
5. Submit pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document complex algorithms

## License

MIT License - See LICENSE file for details