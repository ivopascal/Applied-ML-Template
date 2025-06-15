import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...models.cnn import CNNBackbone
from ...Training.model_trainer import CNNDataset
from ..Evaluation.validation import validate_model

# Add project src to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / 'src'))


def run_evaluation(checkpoint: str, batch_size: int) -> None:
    """
    Evaluate CNN model against a mean-depth baseline on the validation set.

    Args:
        checkpoint: Path to model checkpoint.
        batch_size: Batch size for DataLoader.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load CNN model
    model = CNNBackbone(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # Prepare validation dataset and loader
    val_ds = CNNDataset('val_subset')
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    # Compute global mean ground-truth depth
    all_gt = []
    for _, y in loader:
        all_gt.append(y.numpy().ravel())
    mean_depth = np.concatenate(all_gt).mean()

    # Mean-depth predictor
    class MeanPredictor(torch.nn.Module):
        def __init__(self, m: float) -> None:
            super().__init__()
            self.m = torch.tensor(m, dtype=torch.float32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = x.shape
            return self.m.expand(batch, 1, height, width)

    baseline = MeanPredictor(mean_depth).to(device)

    # Run validation
    metrics_base = validate_model(baseline, val_ds, batch_size, device)
    metrics_cnn = validate_model(model, val_ds, batch_size, device)

    # Print results
    print("\nBaseline (mean-depth) metrics:")
    for name, value in metrics_base.items():
        if name.startswith('delta') or name.startswith('δ'):
            print(f"  {name:6} : {value * 100:5.1f}%")
        else:
            print(f"  {name:6} : {value:7.4f}")

    print("\nCNN-backbone metrics:")
    for name, value in metrics_cnn.items():
        if name.startswith('delta') or name.startswith('δ'):
            print(f"  {name:6} : {value * 100:5.1f}%")
        else:
            print(f"  {name:6} : {value:7.4f}")
