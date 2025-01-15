# Sprite Extractor

A robust, feature-rich tool for extracting individual sprites from sprite sheets. Built with OpenCV and Python, this tool offers both contour-based and grid-based extraction methods, along with various customization options.

## Features

- **Multiple Extraction Modes**
  - Contour-based detection for irregularly spaced sprites
  - Grid-based extraction for uniform sprite sheets
  - Auto-cropping of empty space
  
- **Format Support**
  - PNG output with transparency
  - JPEG with configurable quality
  - WebP support for modern applications
  
- **Advanced Options**
  - Customizable padding around sprites
  - Size filtering (min/max dimensions)
  - Adaptive thresholding
  - JSON metadata generation
  
- **Quality Assurance**
  - Sprite validation
  - Overlap detection
  - Detailed logging
  - Progress tracking
  - Rich console output

## Installation

### Using Poetry (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/ericsonwillians/sprite-extractor.git
cd sprite-extractor
```

2. Install with Poetry:
```bash
poetry install
```

### Manual Installation

1. Ensure you have Python 3.8 or newer installed
2. Install dependencies:
```bash
pip install opencv-python rich numpy
```

## Usage

### Basic Command

```bash
poetry run sprite-extractor input_image.png output_directory/
```

### Advanced Examples

#### Grid-Based Extraction
For sprite sheets with uniform grid layouts:
```bash
poetry run sprite-extractor input.png output/ \
    --grid-mode \
    --grid-size 32 32
```

#### Contour-Based Extraction with Options
For sprite sheets with varying sprite sizes:
```bash
poetry run sprite-extractor input.png output/ \
    --format png \
    --threshold 20 \
    --padding 2 \
    --min-size 10 \
    --max-size 200 \
    --auto-crop \
    --metadata
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `input_image` | Path to input sprite sheet | Required |
| `output_dir` | Directory for extracted sprites | Required |
| `--format` | Output format (png/jpg/webp) | png |
| `--threshold` | Binarization threshold (0-255) | 10 |
| `--padding` | Padding around sprites (pixels) | 0 |
| `--min-size` | Minimum sprite dimension | 5 |
| `--max-size` | Maximum sprite dimension | None |
| `--grid-mode` | Enable grid-based extraction | False |
| `--grid-size` | Grid cell dimensions (WIDTH HEIGHT) | Required with --grid-mode |
| `--auto-crop` | Remove empty space around sprites | False |
| `--metadata` | Generate JSON metadata | False |
| `--debug` | Enable debug output | False |

## Project Structure

```
sprite-extractor/
├── src/
│   └── extractor.py
├── tests/
│   └── test_extractor.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Configuration

### pyproject.toml

The project uses Poetry for dependency management. Here's the default configuration:

```toml
[tool.poetry]
name = "sprite_extractor"
version = "0.1.0"
description = "A tool to extract individual sprites from a sprite sheet into PNG images."
authors = ["Your Name <your.email@example.com>"]
license = "MIT"
readme = "README.md"
packages = [{ include = "src" }]

[tool.poetry.dependencies]
python = "^3.8"
opencv-python = "^4.7.0"
rich = "^13.3.3"
numpy = "^1.24.3"

[tool.poetry.dev-dependencies]
pytest = "^7.2.2"

[tool.poetry.scripts]
sprite-extractor = "src.extractor:main"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

## Output Files

### Sprite Images
Extracted sprites are saved with sequential numbering:
- `sprite_0001.png`
- `sprite_0002.png`
- etc.

### Metadata JSON
When `--metadata` is enabled, generates `sprites_metadata.json`:
```json
{
  "sprites": [
    {
      "index": 1,
      "x": 0,
      "y": 0,
      "width": 32,
      "height": 32,
      "area": 1024,
      "filename": "sprite_0001.png"
    }
  ],
  "total_sprites": 1,
  "extraction_mode": "contour",
  "settings": {
    "threshold": 10,
    "padding": 0,
    "min_size": 5,
    "max_size": null,
    "format": "png",
    "auto_crop": false
  }
}
```

## Logging

The tool maintains detailed logs in `sprite_extraction.log`, including:
- Processing steps
- Warnings and errors
- Extraction statistics
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **No Sprites Detected**
   - Try adjusting the `--threshold` value
   - Check if the sprite sheet has transparency
   - Enable `--debug` for more information

2. **Sprites Are Cut Off**
   - Increase `--padding` value
   - Check `--min-size` and `--max-size` settings
   - Try grid mode if sprites are uniform

3. **Performance Issues**
   - Reduce image size if possible
   - Adjust `--threshold` for faster processing
   - Use grid mode for uniform sprite sheets

### Debug Mode

Enable debug mode for detailed output:
```bash
poetry run sprite-extractor input.png output/ --debug
```

## Performance Tips

1. Use grid mode when possible
2. Optimize input images
3. Adjust threshold values
4. Consider format compression settings
5. Use appropriate min/max size filters