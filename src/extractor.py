#!/usr/bin/env python3

import argparse
import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table
import logging
from pathlib import Path
import json

@dataclass
class SpriteMetadata:
    index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    filename: str

class SpriteExtractor:
    def __init__(self, console: Console):
        self.console = console
        self.logger = self._setup_logger()
        
    @staticmethod
    def _setup_logger():
        logger = logging.getLogger('SpriteExtractor')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler('sprite_extraction.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

    def parse_arguments(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            description="Advanced sprite sheet extractor with multiple processing options.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        # Required arguments
        parser.add_argument(
            "input_image",
            type=str,
            help="Path to the input sprite sheet image."
        )
        parser.add_argument(
            "output_dir",
            type=str,
            help="Directory to save the extracted sprite images."
        )
        
        # Optional arguments
        parser.add_argument(
            "--format",
            type=str,
            choices=['png', 'jpg', 'webp'],
            default='png',
            help="Output format for sprite images."
        )
        parser.add_argument(
            "--threshold",
            type=int,
            default=10,
            help="Threshold value for image binarization (0-255)."
        )
        parser.add_argument(
            "--padding",
            type=int,
            default=0,
            help="Padding around each sprite in pixels."
        )
        parser.add_argument(
            "--min-size",
            type=int,
            default=5,
            help="Minimum sprite size in pixels (both width and height)."
        )
        parser.add_argument(
            "--max-size",
            type=int,
            default=None,
            help="Maximum sprite size in pixels (both width and height)."
        )
        parser.add_argument(
            "--grid-mode",
            action="store_true",
            help="Enable grid-based extraction instead of contour detection."
        )
        parser.add_argument(
            "--grid-size",
            type=int,
            nargs=2,
            metavar=('WIDTH', 'HEIGHT'),
            help="Grid cell size in pixels (requires --grid-mode)."
        )
        parser.add_argument(
            "--auto-crop",
            action="store_true",
            help="Automatically crop empty space around sprites."
        )
        parser.add_argument(
            "--metadata",
            action="store_true",
            help="Generate JSON metadata file with sprite information."
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with additional output."
        )
        
        args = parser.parse_args()
        
        # Validate arguments
        if args.grid_mode and not args.grid_size:
            parser.error("--grid-mode requires --grid-size")
            
        if args.threshold < 0 or args.threshold > 255:
            parser.error("--threshold must be between 0 and 255")
            
        return args

    def create_output_directory(self, path: str) -> Path:
        output_path = Path(path)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            self.console.log(f"[green]Output directory ready: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"Failed to create directory {path}: {e}")
            raise

    def load_image(self, path: str) -> np.ndarray:
        image_path = Path(path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
            
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {path}")
            
        self.console.log(f"[green]Loaded image: {path}")
        self.console.log(f"Shape: {image.shape}, Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
        return image

    def preprocess_image(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """Enhanced preprocessing optimized for sprite sheets with complex shapes"""
        if len(image.shape) == 2:  # Grayscale
            alpha = image
        elif image.shape[2] == 4:  # RGBA
            # Use alpha channel but also consider RGB values for semi-transparent pixels
            alpha = image[:, :, 3]
            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            # Combine alpha with RGB information
            alpha = cv2.addWeighted(alpha, 0.7, rgb, 0.3, 0)
        elif image.shape[2] == 3:  # RGB
            alpha = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported image format with {image.shape[2]} channels")

        # Normalize alpha channel
        alpha = cv2.normalize(alpha, None, 0, 255, cv2.NORM_MINMAX)

        # Apply bilateral filter to reduce noise while preserving edges
        alpha = cv2.bilateralFilter(alpha, 9, 75, 75)

        # Create binary mask using Otsu's method
        _, binary = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Enhance edges
        kernel_edge = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel_edge)

        # Clean up noise
        kernel_clean = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)

        return binary

    def find_sprites_contour_mode(
        self,
        binary: np.ndarray,
        min_size: int,
        max_size: Optional[int]
    ) -> List[np.ndarray]:
        """Enhanced sprite detection optimized for complex weapon sprites"""
        # First pass: Find all potential contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,  # Use TREE to get hierarchy information
            cv2.CHAIN_APPROX_TC89_KCOS  # More precise approximation
        )
        
        # Process contours with hierarchy information
        filtered_contours = []
        if len(contours) > 0 and hierarchy is not None:
            hierarchy = hierarchy[0]  # Unwrap hierarchy array
            
            for idx, (contour, hier) in enumerate(zip(contours, hierarchy)):
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Skip if too small or too large
                if w < min_size or h < min_size:
                    continue
                if max_size and (w > max_size or h > max_size):
                    continue
                
                # Calculate contour properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # Calculate shape complexity
                complexity = len(cv2.approxPolyDP(contour, 0.02 * perimeter, True))
                
                # Calculate solidity
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Filtering criteria for weapon sprites
                is_valid = (
                    # Has reasonable area relative to bounding box
                    area > 0.15 * w * h and
                    # Not too simple (likely not a weapon if too simple)
                    complexity >= 4 and
                    # Not too complex (likely noise if too complex)
                    complexity <= 100 and
                    # Has reasonable solidity (weapons are typically solid shapes)
                    solidity > 0.2 and
                    # Has reasonable aspect ratio for weapons
                    0.1 <= w / h <= 10
                )
                
                if is_valid:
                    # Check if this contour might be part of a larger weapon
                    parent_idx = hier[3]
                    if parent_idx != -1:
                        parent_area = cv2.contourArea(contours[parent_idx])
                        # If parent is significantly larger, skip this contour
                        if parent_area > area * 1.5:
                            continue
                    
                    # Add contour with additional information for post-processing
                    filtered_contours.append({
                        'contour': contour,
                        'area': area,
                        'complexity': complexity,
                        'solidity': solidity,
                        'bounds': (x, y, w, h)
                    })
        
        # Sort contours by area (largest first)
        filtered_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Post-process to remove redundant overlapping contours
        final_contours = []
        used_areas = set()
        
        for cont_info in filtered_contours:
            x, y, w, h = cont_info['bounds']
            area_key = f"{x//5}_{y//5}_{w//5}_{h//5}"  # Group similar areas
            
            if area_key not in used_areas:
                used_areas.add(area_key)
                final_contours.append(cont_info['contour'])
        
        return final_contours

    def find_sprites_grid_mode(
        self,
        image: np.ndarray,
        grid_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """Find sprites using grid-based detection"""
        height, width = image.shape[:2]
        grid_w, grid_h = grid_size
        
        sprites = []
        for y in range(0, height - grid_h + 1, grid_h):
            for x in range(0, width - grid_w + 1, grid_w):
                # Check if grid cell contains non-transparent pixels
                cell = image[y:y+grid_h, x:x+grid_w]
                if len(image.shape) == 4:  # RGBA
                    if np.any(cell[:, :, 3] > 0):
                        sprites.append((x, y, grid_w, grid_h))
                else:  # RGB
                    if np.any(cell > 0):
                        sprites.append((x, y, grid_w, grid_h))
                        
        return sprites

    def auto_crop_sprite(self, sprite: np.ndarray) -> np.ndarray:
        """Automatically crop empty space around sprite"""
        if len(sprite.shape) == 4:  # RGBA
            mask = sprite[:, :, 3] > 0
        else:  # RGB
            mask = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY) > 0
            
        if not np.any(mask):  # If sprite is empty
            return sprite
            
        # Find bounds of non-empty region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        return sprite[ymin:ymax+1, xmin:xmax+1]

    def save_sprite(
        self,
        sprite: np.ndarray,
        filename: str,
        format: str
    ) -> bool:
        """Save sprite with format handling"""
        try:
            if format == 'png':
                cv2.imwrite(filename, sprite, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            elif format == 'jpg':
                cv2.imwrite(filename, sprite, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif format == 'webp':
                cv2.imwrite(filename, sprite, [cv2.IMWRITE_WEBP_QUALITY, 95])
            return True
        except Exception as e:
            self.logger.error(f"Failed to save sprite {filename}: {e}")
            return False

    def extract_sprites(self, args: argparse.Namespace) -> List[SpriteMetadata]:
        """Main sprite extraction process"""
        # Load and preprocess image
        image = self.load_image(args.input_image)
        binary = self.preprocess_image(image, args.threshold)
        
        sprites_metadata = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            if args.grid_mode:
                sprite_regions = self.find_sprites_grid_mode(image, args.grid_size)
                task = progress.add_task("Extracting sprites (grid mode)...", total=len(sprite_regions))
                
                for idx, (x, y, w, h) in enumerate(sprite_regions, 1):
                    sprite = image[y:y+h, x:x+w]
                    if args.auto_crop:
                        sprite = self.auto_crop_sprite(sprite)
                        
                    filename = f"sprite_{idx:04d}.{args.format}"
                    filepath = Path(args.output_dir) / filename
                    
                    if self.save_sprite(sprite, str(filepath), args.format):
                        sprites_metadata.append(SpriteMetadata(
                            index=idx,
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            area=w*h,
                            filename=filename
                        ))
                    
                    progress.update(task, advance=1)
                    
            else:  # Contour mode
                contours = self.find_sprites_contour_mode(binary, args.min_size, args.max_size)
                task = progress.add_task("Extracting sprites (contour mode)...", total=len(contours))
                
                for idx, contour in enumerate(contours, 1):
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Apply padding
                    x_pad = max(x - args.padding, 0)
                    y_pad = max(y - args.padding, 0)
                    w_pad = min(w + 2 * args.padding, image.shape[1] - x_pad)
                    h_pad = min(h + 2 * args.padding, image.shape[0] - y_pad)
                    
                    sprite = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                    
                    if args.auto_crop:
                        sprite = self.auto_crop_sprite(sprite)
                    
                    filename = f"sprite_{idx:04d}.{args.format}"
                    filepath = Path(args.output_dir) / filename
                    
                    if self.save_sprite(sprite, str(filepath), args.format):
                        sprites_metadata.append(SpriteMetadata(
                            index=idx,
                            x=x,
                            y=y,
                            width=w,
                            height=h,
                            area=cv2.contourArea(contour),
                            filename=filename
                        ))
                    
                    progress.update(task, advance=1)
        
        return sprites_metadata

    def save_metadata(self, metadata: List[SpriteMetadata], output_dir: str, settings: argparse.Namespace):
        """Save sprite metadata to JSON file"""
        metadata_file = Path(output_dir) / "sprites_metadata.json"
        metadata_dict = {
            "sprites": [vars(sprite) for sprite in metadata],
            "total_sprites": len(metadata),
            "extraction_mode": "grid" if settings.grid_mode else "contour",
            "settings": {
                "threshold": settings.threshold,
                "padding": settings.padding,
                "min_size": settings.min_size,
                "max_size": settings.max_size,
                "format": settings.format,
                "auto_crop": settings.auto_crop
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
            
        self.console.log(f"[green]Saved metadata to {metadata_file}")

    def print_summary(self, metadata: List[SpriteMetadata]):
        """Print extraction summary"""
        table = Table(title="Sprite Extraction Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total sprites", str(len(metadata)))
        table.add_row("Average width", f"{sum(s.width for s in metadata) / len(metadata):.1f}px")
        table.add_row("Average height", f"{sum(s.height for s in metadata) / len(metadata):.1f}px")
        
        if metadata:
            min_sprite = min(metadata, key=lambda s: s.area)
            max_sprite = max(metadata, key=lambda s: s.area)
            table.add_row("Smallest sprite", f"{min_sprite.width}x{min_sprite.height}px")
            table.add_row("Largest sprite", f"{max_sprite.width}x{max_sprite.height}px")
        
        self.console.print(table)
        
    def validate_sprites(self, metadata: List[SpriteMetadata]) -> bool:
        """Validate extracted sprites for potential issues"""
        if not metadata:
            self.console.print("[red]Warning: No sprites were extracted!")
            return False
            
        issues = []
        
        # Check for overlapping sprites
        for i, sprite1 in enumerate(metadata):
            for sprite2 in metadata[i+1:]:
                rect1 = (sprite1.x, sprite1.y, sprite1.x + sprite1.width, sprite1.y + sprite1.height)
                rect2 = (sprite2.x, sprite2.y, sprite2.x + sprite2.width, sprite2.y + sprite2.height)
                
                if (rect1[0] < rect2[2] and rect1[2] > rect2[0] and
                    rect1[1] < rect2[3] and rect1[3] > rect2[1]):
                    issues.append(f"Sprites {sprite1.index} and {sprite2.index} overlap")
        
        # Check for unusually small or large sprites
        sizes = [s.area for s in metadata]
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        for sprite in metadata:
            if abs(sprite.area - mean_size) > 3 * std_size:
                issues.append(f"Sprite {sprite.index} has unusual size: {sprite.width}x{sprite.height}")
        
        if issues:
            self.console.print(Panel(
                "\n".join(issues),
                title="[yellow]Sprite Validation Issues",
                border_style="yellow"
            ))
            return False
            
        return True

def main():
    console = Console()
    
    try:
        extractor = SpriteExtractor(console)
        args = extractor.parse_arguments()
        
        # Create output directory
        output_dir = extractor.create_output_directory(args.output_dir)
        
        # Extract sprites
        console.print("[blue]Starting sprite extraction...")
        sprites_metadata = extractor.extract_sprites(args)
        
        # Validate results
        if extractor.validate_sprites(sprites_metadata):
            console.print("[green]Sprite validation passed!")
        
        # Save metadata if requested
        if args.metadata:
            extractor.save_metadata(sprites_metadata, output_dir, args)
        
        # Print summary
        extractor.print_summary(sprites_metadata)
        
        console.print(f"[green]Successfully extracted {len(sprites_metadata)} sprites!")
        
    except Exception as e:
        console.print(f"[red]Error during sprite extraction: {str(e)}")
        logging.exception("Unhandled exception during sprite extraction")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())