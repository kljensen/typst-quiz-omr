#!/usr/bin/env python3
"""
CLI for OMR detection and testing
"""

import click
from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
import logging
from typing import Optional

from detector import DynamicOMRDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def pdf_to_images(pdf_path: Path, output_dir: Optional[Path] = None):
    """Convert PDF pages to images"""
    if output_dir is None:
        output_dir = Path("tests/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    images = convert_from_path(pdf_path, dpi=300)
    image_paths = []
    
    for i, image in enumerate(images):
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Save as PNG
        output_path = output_dir / f"{pdf_path.stem}_page_{i+1}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        image_paths.append(output_path)
        logger.info(f"Saved {output_path}")
    
    return image_paths


def visualize_detection(image_path: Path, detector: DynamicOMRDetector, output_path: Optional[Path] = None):
    """Visualize the detection results"""
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find bubble region
    region = detector.find_bubble_region(gray)
    
    if region:
        x, y, w, h = region
        # Draw rectangle around bubble region
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract region and detect bubbles
        extracted = gray[y:y+h, x:x+w]
        bubbles = detector.detect_bubbles(extracted)
        
        # Draw detected bubbles
        for bubble in bubbles:
            cx, cy = bubble.center
            # Adjust coordinates to full image
            cx += x
            cy += y
            cv2.circle(img, (cx, cy), int(bubble.radius), (255, 0, 0), 2)
    
    # Save or show result
    if output_path:
        cv2.imwrite(str(output_path), img)
        logger.info(f"Visualization saved to {output_path}")
    else:
        # Display (might not work in all environments)
        cv2.imshow("Detection Result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return img


@click.group()
def cli():
    """OMR Detection CLI"""
    pass


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', type=click.Path(path_type=Path), help='Output directory for images')
def convert(pdf_path: Path, output_dir: Optional[Path]):
    """Convert PDF to images for OMR processing"""
    logger.info(f"Converting {pdf_path} to images...")
    image_paths = pdf_to_images(pdf_path, output_dir)
    logger.info(f"Converted {len(image_paths)} pages")
    return image_paths


@cli.command()
@click.argument('image_path', type=click.Path(exists=True, path_type=Path))
@click.option('--visualize', is_flag=True, help='Save visualization of detection')
@click.option('--output', type=click.Path(path_type=Path), help='Output path for visualization')
def detect(image_path: Path, visualize: bool, output: Optional[Path]):
    """Detect and read OMR bubbles from an image"""
    detector = DynamicOMRDetector()
    
    try:
        # Run detection
        answers = detector.process(image_path)
        
        # Print results
        logger.info(f"Detected answers for {len(answers)} questions:")
        for q_num in sorted(answers.keys()):
            answer = answers[q_num] or "blank"
            print(f"  Question {q_num}: {answer}")
        
        # Visualize if requested
        if visualize:
            if output is None:
                output = image_path.parent / f"{image_path.stem}_detected.png"
            visualize_detection(image_path, detector, output)
            
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True, path_type=Path))
@click.option('--visualize', is_flag=True, help='Save visualizations')
def process_pdf(pdf_path: Path, visualize: bool):
    """Process all pages of a PDF"""
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Convert to images
    image_paths = pdf_to_images(pdf_path)
    
    # Process each page
    detector = DynamicOMRDetector()
    
    for i, img_path in enumerate(image_paths):
        logger.info(f"\nProcessing page {i+1}...")
        try:
            answers = detector.process(img_path)
            
            print(f"\nPage {i+1} Results:")
            for q_num in sorted(answers.keys()):
                answer = answers[q_num] or "blank"
                print(f"  Question {q_num}: {answer}")
                
            if visualize:
                output = img_path.parent / f"{img_path.stem}_detected.png"
                visualize_detection(img_path, detector, output)
                
        except Exception as e:
            logger.error(f"Failed to process page {i+1}: {e}")


@cli.command()
def test():
    """Run tests on generated test data"""
    test_pdf = Path("test_data.pdf")
    
    if not test_pdf.exists():
        logger.error("test_data.pdf not found. Run: typst compile generate_test_data.typ test_data.pdf")
        return
    
    # Expected answers for our test cases
    expected = {
        1: {  # 5 questions sequential
            1: "A", 2: "B", 3: "C", 4: "D", 5: "E"
        },
        2: {  # 10 questions mixed
            1: "C", 2: "C", 3: "A", 4: "B", 5: "D",
            6: "E", 7: "A", 8: "B", 9: "C", 10: "D"
        },
        3: {  # 15 questions partial
            1: "A", 3: "B", 5: "C", 7: "D", 9: "E",
            11: "A", 13: "B", 15: "C"
        },
        4: {},  # 20 questions all blank
        5: {  # 7 questions all A
            1: "A", 2: "A", 3: "A", 4: "A", 5: "A", 6: "A", 7: "A"
        }
    }
    
    # Convert and process
    logger.info("Converting test PDF to images...")
    image_paths = pdf_to_images(test_pdf)
    
    detector = DynamicOMRDetector()
    results = []
    
    for i, img_path in enumerate(image_paths):
        if i >= 5:  # Skip pages after test cases
            break
            
        logger.info(f"\nTesting case {i+1}...")
        try:
            answers = detector.process(img_path)
            results.append(answers)
            
            # Compare with expected
            test_expected = expected[i+1]
            
            correct = 0
            total = max(len(answers), len(test_expected))
            
            for q_num in range(1, total + 1):
                detected = answers.get(q_num)
                expected_ans = test_expected.get(q_num)
                
                if detected == expected_ans:
                    correct += 1
                else:
                    logger.warning(f"  Q{q_num}: Expected {expected_ans}, got {detected}")
                    
            accuracy = (correct / total * 100) if total > 0 else 0
            logger.info(f"  Accuracy: {correct}/{total} = {accuracy:.1f}%")
            
        except Exception as e:
            logger.error(f"Test case {i+1} failed: {e}")
            results.append(None)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    logger.info(f"\n=== Test Summary ===")
    logger.info(f"Successful: {successful}/{len(results)}")


if __name__ == "__main__":
    cli()