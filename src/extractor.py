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
        """Enhanced sprite detection with duplicate removal"""
        # First pass: Find all potential contours
        contours, hierarchy = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_TC89_KCOS
        )
        
        # Process contours with hierarchy information
        filtered_contours = []
        if len(contours) > 0 and hierarchy is not None:
            hierarchy = hierarchy[0]
            
            for idx, (contour, hier) in enumerate(zip(contours, hierarchy)):
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Basic size filtering
                if w < min_size or h < min_size:
                    continue
                if max_size and (w > max_size or h > max_size):
                    continue
                
                # Calculate shape properties
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                complexity = len(cv2.approxPolyDP(contour, 0.02 * perimeter, True))
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Extract the sprite region for content-based comparison
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
                sprite_content = cv2.bitwise_and(binary, mask)
                sprite_region = sprite_content[y:y+h, x:x+w]
                
                # Calculate content hash for duplicate detection
                content_hash = hash(sprite_region.tobytes())
                
                contour_info = {
                    'contour': contour,
                    'area': area,
                    'complexity': complexity,
                    'solidity': solidity,
                    'bounds': (x, y, w, h),
                    'content_hash': content_hash,
                    'sprite_region': sprite_region
                }
                
                if self._is_valid_sprite(contour_info, hier, contours):
                    filtered_contours.append(contour_info)
        
        # Sort contours by area (largest first)
        filtered_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Remove duplicates using multiple criteria
        final_contours = self._remove_duplicates(filtered_contours)
        
        return [cont_info['contour'] for cont_info in final_contours]

    def _is_valid_sprite(self, contour_info: dict, hierarchy: np.ndarray, all_contours: List[np.ndarray]) -> bool:
        """Validate if a contour represents a valid sprite"""
        area = contour_info['area']
        complexity = contour_info['complexity']
        solidity = contour_info['solidity']
        x, y, w, h = contour_info['bounds']
        
        # Basic shape validation
        is_valid = (
            area > 0.15 * w * h and
            complexity >= 4 and
            complexity <= 100 and
            solidity > 0.2 and
            0.1 <= w / h <= 10
        )
        
        if not is_valid:
            return False
        
        # Check hierarchy relationships
        parent_idx = hierarchy[3]
        if parent_idx != -1:
            parent_area = cv2.contourArea(all_contours[parent_idx])
            if parent_area > area * 1.5:
                return False
        
        return True

    def _remove_duplicates(self, contours_info: List[dict]) -> List[dict]:
        """Remove duplicate sprites using multiple criteria"""
        unique_contours = []
        used_hashes = set()
        
        for cont_info in contours_info:
            x, y, w, h = cont_info['bounds']
            content_hash = cont_info['content_hash']
            
            # Skip if we've seen this exact content before
            if content_hash in used_hashes:
                continue
            
            # Check for similar sprites
            is_duplicate = False
            for unique_cont in unique_contours:
                ux, uy, uw, uh = unique_cont['bounds']
                
                # Check for significant overlap
                overlap_x = max(0, min(x + w, ux + uw) - max(x, ux))
                overlap_y = max(0, min(y + h, uy + uh) - max(y, uy))
                overlap_area = overlap_x * overlap_y
                min_area = min(w * h, uw * uh)
                
                if overlap_area > 0.8 * min_area:  # 80% overlap threshold
                    # Compare actual content
                    if self._compare_sprite_content(cont_info['sprite_region'], 
                                                 unique_cont['sprite_region']):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_contours.append(cont_info)
                used_hashes.add(content_hash)
        
        return unique_contours

    def _compare_sprite_content(self, sprite1: np.ndarray, sprite2: np.ndarray) -> bool:
        """Compare two sprite regions for similarity"""
        # Resize to same dimensions for comparison
        if sprite1.shape != sprite2.shape:
            h1, w1 = sprite1.shape
            h2, w2 = sprite2.shape
            max_h, max_w = max(h1, h2), max(w1, w2)
            sprite1_resized = cv2.resize(sprite1, (max_w, max_h))
            sprite2_resized = cv2.resize(sprite2, (max_w, max_h))
        else:
            sprite1_resized = sprite1
            sprite2_resized = sprite2
        
        # Calculate similarity
        diff = cv2.absdiff(sprite1_resized, sprite2_resized)
        similarity = 1.0 - (np.count_nonzero(diff) / diff.size)
        
        return similarity > 0.9  # 90% similarity threshold

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
        """Main sprite extraction process with professional duplicate handling"""
        # Load and preprocess image
        image = self.load_image(args.input_image)
        binary = self.preprocess_image(image, args.threshold)
        
        # Store extracted sprites with their data for duplicate analysis
        extracted_sprites = []
        
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
                    sprite_data = self._extract_sprite_data(image, x, y, w, h, args)
                    if sprite_data:
                        extracted_sprites.append(sprite_data)
                    progress.update(task, advance=1)
            else:
                contours = self.find_sprites_contour_mode(binary, args.min_size, args.max_size)
                task = progress.add_task("Extracting sprites (contour mode)...", total=len(contours))
                
                for idx, contour in enumerate(contours, 1):
                    x, y, w, h = cv2.boundingRect(contour)
                    sprite_data = self._extract_sprite_data(image, x, y, w, h, args)
                    if sprite_data:
                        extracted_sprites.append(sprite_data)
                    progress.update(task, advance=1)

        # Professional duplicate removal process
        unique_sprites = self._remove_duplicates_professional(extracted_sprites)
        
        # Save unique sprites
        sprites_metadata = []
        for idx, sprite_data in enumerate(unique_sprites, 1):
            filename = f"sprite_{idx:04d}.{args.format}"
            filepath = Path(args.output_dir) / filename
            
            if self.save_sprite(sprite_data['sprite'], str(filepath), args.format):
                sprites_metadata.append(SpriteMetadata(
                    index=idx,
                    x=sprite_data['x'],
                    y=sprite_data['y'],
                    width=sprite_data['width'],
                    height=sprite_data['height'],
                    area=sprite_data['area'],
                    filename=filename
                ))
        
        return sprites_metadata

    def _extract_sprite_data(self, image: np.ndarray, x: int, y: int, w: int, h: int, args: argparse.Namespace) -> Optional[dict]:
        """Extract sprite data with advanced preprocessing"""
        try:
            # Apply padding
            x_pad = max(x - args.padding, 0)
            y_pad = max(y - args.padding, 0)
            w_pad = min(w + 2 * args.padding, image.shape[1] - x_pad)
            h_pad = min(h + 2 * args.padding, image.shape[0] - y_pad)
            
            sprite = image[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad].copy()
            
            if args.auto_crop:
                sprite = self.auto_crop_sprite(sprite)
            
            if sprite.size == 0:
                return None
                
            # Calculate sprite signature
            signature = self._calculate_sprite_signature(sprite)
            
            return {
                'sprite': sprite,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'area': w * h,
                'signature': signature,
                'bounds': (x, y, w, h)
            }
        except Exception as e:
            self.logger.error(f"Error extracting sprite at ({x}, {y}): {e}")
            return None

    def _calculate_sprite_signature(self, sprite: np.ndarray) -> dict:
        """Calculate comprehensive sprite signature for accurate duplicate detection"""
        # Convert to grayscale if needed
        if len(sprite.shape) > 2:
            if sprite.shape[2] == 4:  # RGBA
                # Use alpha channel and RGB information
                alpha = sprite[:, :, 3]
                rgb = cv2.cvtColor(sprite[:, :, :3], cv2.COLOR_BGR2GRAY)
                gray = cv2.addWeighted(alpha, 0.7, rgb, 0.3, 0)
            else:  # RGB
                gray = cv2.cvtColor(sprite, cv2.COLOR_BGR2GRAY)
        else:
            gray = sprite.copy()

        # Normalize size for consistent comparison
        normalized = cv2.resize(gray, (64, 64))
        
        # Calculate multiple signatures for robust comparison
        signatures = {
            # Perceptual hash (robust to minor variations)
            'phash': self._calculate_phash(normalized),
            
            # Histogram signature
            'histogram': cv2.calcHist([normalized], [0], None, [16], [0, 256]).flatten(),
            
            # Edge signature using Canny
            'edges': cv2.Canny(normalized, 100, 200).sum(),
            
            # Shape moments
            'moments': cv2.HuMoments(cv2.moments(normalized)).flatten(),
            
            # Non-zero pixels pattern
            'pattern': (normalized > 0).astype(np.uint8).tobytes(),
            
            # Original dimensions
            'dimensions': sprite.shape[:2]
        }
        
        return signatures

    def _calculate_phash(self, image: np.ndarray, hash_size: int = 8) -> bytes:
        """Calculate perceptual hash of the image"""
        # Resize to 64x64
        resized = cv2.resize(image, (64, 64))
        
        # Calculate DCT
        dct = cv2.dct(np.float32(resized))
        
        # Extract top-left 8x8 DCT coefficients
        dct_low = dct[:hash_size, :hash_size]
        
        # Calculate median value
        med = np.median(dct_low)
        
        # Create hash
        return (dct_low > med).tobytes()

    def _remove_duplicates_professional(self, sprites: List[dict]) -> List[dict]:
        """Professional-grade duplicate removal using multiple criteria"""
        if not sprites:
            return []
            
        unique_sprites = []
        num_sprites = len(sprites)
        
        # Create similarity matrix
        similarity_matrix = np.zeros((num_sprites, num_sprites))
        
        # Calculate similarity scores
        for i in range(num_sprites):
            for j in range(i + 1, num_sprites):
                similarity = self._calculate_similarity_score(
                    sprites[i]['signature'],
                    sprites[j]['signature']
                )
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        # Track processed sprites
        processed = set()
        
        # Group similar sprites
        for i in range(num_sprites):
            if i in processed:
                continue
                
            # Find all sprites similar to current sprite
            similar_indices = np.where(similarity_matrix[i] > 0.85)[0]
            
            if len(similar_indices) > 0:
                # Get the best quality sprite from the group
                group_sprites = [sprites[idx] for idx in similar_indices]
                best_sprite = self._select_best_sprite(group_sprites)
                unique_sprites.append(best_sprite)
                
                # Mark all similar sprites as processed
                processed.update(similar_indices)
            else:
                unique_sprites.append(sprites[i])
                processed.add(i)
        
        return unique_sprites

    def _calculate_similarity_score(self, sig1: dict, sig2: dict) -> float:
        """Calculate comprehensive similarity score between two sprites"""
        # Initialize weights for different features
        weights = {
            'phash': 0.4,
            'histogram': 0.2,
            'edges': 0.1,
            'moments': 0.2,
            'dimensions': 0.1
        }
        
        scores = []
        
        # Compare perceptual hashes
        phash_sim = np.count_nonzero(np.frombuffer(sig1['phash'], dtype=np.bool_) == 
                                   np.frombuffer(sig2['phash'], dtype=np.bool_)) / (8 * 8)
        scores.append(('phash', phash_sim))
        
        # Compare histograms using correlation
        hist_sim = cv2.compareHist(
            np.float32(sig1['histogram']).reshape(-1, 1),
            np.float32(sig2['histogram']).reshape(-1, 1),
            cv2.HISTCMP_CORREL
        )
        scores.append(('histogram', max(0, hist_sim)))
        
        # Compare edge signatures
        edge_sim = 1 - min(1, abs(sig1['edges'] - sig2['edges']) / max(sig1['edges'], sig2['edges']))
        scores.append(('edges', edge_sim))
        
        # Compare moments
        moments_sim = 1 - min(1, np.linalg.norm(sig1['moments'] - sig2['moments']))
        scores.append(('moments', moments_sim))
        
        # Compare dimensions
        dim_sim = 1 - min(1, abs(np.prod(sig1['dimensions']) - np.prod(sig2['dimensions'])) / 
                         max(np.prod(sig1['dimensions']), np.prod(sig2['dimensions'])))
        scores.append(('dimensions', dim_sim))
        
        # Calculate weighted average
        final_score = sum(weights[name] * score for name, score in scores)
        
        return final_score

    def _select_best_sprite(self, sprites: List[dict]) -> dict:
        """Select the best quality sprite from a group of similar sprites"""
        if len(sprites) == 1:
            return sprites[0]
            
        # Score each sprite based on quality metrics
        scored_sprites = []
        for sprite in sprites:
            # Calculate quality score based on multiple factors
            score = 0.0
            
            # Factor 1: Area (prefer larger sprites)
            area = sprite['area']
            score += 0.4 * (area / max(s['area'] for s in sprites))
            
            # Factor 2: Edge clarity
            edges = cv2.Canny(sprite['sprite'], 100, 200)
            edge_density = np.count_nonzero(edges) / area
            score += 0.3 * edge_density
            
            # Factor 3: Position (prefer sprites closer to top-left)
            x, y = sprite['x'], sprite['y']
            position_score = 1 - (np.sqrt(x*x + y*y) / np.sqrt(480*480 + 480*480))
            score += 0.3 * position_score
            
            scored_sprites.append((score, sprite))
        
        # Return sprite with highest score
        return max(scored_sprites, key=lambda x: x[0])[1]

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