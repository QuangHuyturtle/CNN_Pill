"""
Inference script for Pill Image Classification
Supports single image prediction and batch processing
"""

import os
import sys
import argparse
from typing import List, Tuple, Dict, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.efficientnet_pill import create_model


class PillClassifier:
    """
    Pill image classifier for inference.

    Args:
        checkpoint_path: Path to trained model checkpoint
        label_encoder_path: Path to label encoder pickle file
        device: Device to run inference on (cuda/cpu)
        model_size: EfficientNetV2 size used during training
    """

    def __init__(
        self,
        checkpoint_path: str,
        label_encoder_path: str,
        device: Optional[str] = None,
        model_size: str = 's'
    ):
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load checkpoint first to get num_classes
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Get num_classes from checkpoint (fold-specific), or fall back to label encoder
        if 'num_classes' in checkpoint:
            self.num_classes = checkpoint['num_classes']
            print(f"Using num_classes from checkpoint: {self.num_classes}")
        else:
            # Fallback: load label encoder to get num_classes
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
            self.num_classes = len(label_encoder)
            print(f"Warning: checkpoint doesn't have num_classes, using label encoder: {self.num_classes}")

        # Create model with correct num_classes
        self.model = create_model(
            num_classes=self.num_classes,
            model_size=model_size,
            pretrained=False
        )

        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        if 'best_acc1' in checkpoint:
            print(f"Checkpoint Top-1 Accuracy: {checkpoint['best_acc1']:.2f}%")
        if 'fold_idx' in checkpoint:
            print(f"Checkpoint Fold: {checkpoint['fold_idx']}")

        # Load label encoder for inference
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)

        self.idx_to_label = {v: k for k, v in label_encoder.items()}

        # Image normalization (ImageNet stats)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.image_size = 224

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for model input.

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed tensor ready for model input
        """
        # Load image
        image = Image.open(image_path).convert('RGB')

        # Resize
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        # Convert to tensor
        image = torch.from_numpy(np.array(image)).float() / 255.0
        image = image.permute(2, 0, 1)  # HWC -> CHW

        # Normalize
        for i in range(3):
            image[i] = (image[i] - self.mean[i]) / self.std[i]

        # Add batch dimension
        image = image.unsqueeze(0)

        return image.to(self.device)

    def predict(
        self,
        image_path: str,
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Predict pill class from image.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            List of dictionaries with label, confidence, and rank
        """
        # Preprocess
        image_tensor = self.preprocess_image(image_path)

        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get top-k predictions
        probs, indices = torch.topk(probabilities, min(top_k, self.num_classes))

        # Format results
        results = []
        for i, (idx, prob) in enumerate(zip(indices[0], probs[0])):
            label = self.idx_to_label[idx.item()]
            results.append({
                'rank': i + 1,
                'label': label,
                'confidence': prob.item() * 100
            })

        return results

    def predict_batch(
        self,
        image_paths: List[str],
        top_k: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Predict multiple images.

        Args:
            image_paths: List of image paths
            top_k: Number of top predictions per image

        Returns:
            Dictionary mapping image path to predictions
        """
        results = {}

        for image_path in image_paths:
            try:
                predictions = self.predict(image_path, top_k)
                results[image_path] = predictions
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results[image_path] = None

        return results

    def visualize_prediction(
        self,
        image_path: str,
        predictions: List[Dict],
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction results on the image.

        Args:
            image_path: Path to original image
            predictions: Prediction results from predict()
            save_path: Optional path to save visualization
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            # Try PIL if cv2 fails
            image_pil = Image.open(image_path).convert('RGB')
            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]

        # Create overlay for predictions
        overlay_height = 30 + len(predictions) * 30
        overlay = np.zeros((overlay_height, w, 3), dtype=np.uint8)
        overlay[:] = (40, 40, 40)

        # Add title
        cv2.putText(overlay, "Top Predictions:", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Add predictions
        for i, pred in enumerate(predictions[:5]):
            y = 50 + i * 30
            text = f"{i+1}. {pred['label'][:30]}... ({pred['confidence']:.1f}%)"
            cv2.putText(overlay, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Color bar for confidence
            bar_width = int(w * pred['confidence'] / 100 * 0.3)
            color = (0, int(255 * pred['confidence'] / 100), int(255 * (1 - pred['confidence'] / 100)))
            cv2.rectangle(overlay, (w - bar_width - 10, y - 15),
                         (w - 10, y - 5), color, -1)

        # Combine image and overlay
        result = np.vstack([overlay, image])

        if save_path:
            cv2.imwrite(save_path, result)
            print(f"Visualization saved to {save_path}")
        else:
            # Display using cv2 (may not work in all environments)
            try:
                cv2.imshow('Prediction', result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except:
                print("Cannot display image. Try using save_path instead.")

    def predict_directory(
        self,
        directory: str,
        output_csv: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict all images in a directory.

        Args:
            directory: Path to directory containing images
            output_csv: Optional path to save results as CSV
            top_k: Number of top predictions per image

        Returns:
            List of prediction results
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(Path(directory).glob(f"*{ext}"))
            image_paths.extend(Path(directory).glob(f"*{ext.upper()}"))

        if not image_paths:
            print(f"No images found in {directory}")
            return []

        print(f"Found {len(image_paths)} images")

        results = []
        for image_path in image_paths:
            try:
                predictions = self.predict(str(image_path), top_k)
                results.append({
                    'image': str(image_path),
                    'predictions': predictions
                })
                print(f"[{results.index(results[-1])+1}/{len(image_paths)}] {image_path.name}: {predictions[0]['label']}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        if output_csv:
            import csv
            with open(output_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Rank', 'Label', 'Confidence'])

                for result in results:
                    image_name = Path(result['image']).name
                    for pred in result['predictions']:
                        writer.writerow([
                            image_name,
                            pred['rank'],
                            pred['label'],
                            f"{pred['confidence']:.2f}%"
                        ])

            print(f"\nResults saved to {output_csv}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Pill Image Classification Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--encoder', type=str,
                        default='data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/label_encoder.pickle',
                        help='Path to label encoder pickle')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for prediction')
    parser.add_argument('--directory', type=str, default=None,
                        help='Path to directory for batch prediction')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV path for batch results')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top predictions to return')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization overlays')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['s', 'm', 'l'],
                        help='Model size used during training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detect if not specified')

    args = parser.parse_args()

    # Create classifier
    classifier = PillClassifier(
        checkpoint_path=args.checkpoint,
        label_encoder_path=args.encoder,
        device=args.device,
        model_size=args.model_size
    )

    # Run prediction
    if args.image:
        print(f"\nPredicting: {args.image}")
        print("-" * 60)

        predictions = classifier.predict(args.image, top_k=args.top_k)

        for pred in predictions:
            print(f"  {pred['rank']}. {pred['label']}")
            print(f"     Confidence: {pred['confidence']:.2f}%")

        if args.visualize:
            viz_path = args.output or args.image.replace('.', '_pred.')
            classifier.visualize_prediction(args.image, predictions, save_path=viz_path)

    elif args.directory:
        print(f"\nPredicting all images in: {args.directory}")
        print("=" * 60)

        results = classifier.predict_directory(
            directory=args.directory,
            output_csv=args.output,
            top_k=args.top_k
        )

        print(f"\nCompleted {len(results)} predictions")

    else:
        print("Please specify --image or --directory")
        parser.print_help()


if __name__ == '__main__':
    main()
