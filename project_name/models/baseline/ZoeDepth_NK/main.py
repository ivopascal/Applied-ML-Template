import torch  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from .data_loader import ZoeDepthDataset
from .eval import evaluate_zoedepth_model


class ZoeDepthEvaluator:
    """Zoedepth model
    """
    def __init__(
        self,
        split: str = "val",
        batch_size: int = 1,
        num_workers: int = 4
    ) -> None:
        """init the model

        Args:
            split (str, optional): mode. Defaults to "val".
            batch_size (int, optional): batch size. Defaults to 1.
            num_workers (int, optional): amount worker. Defaults to 4.
        """
        d = torch.device("cuda"if torch.cuda.is_available() else "cpu")
        self.device = d

        self.split = split
        self.batch_size = batch_size
        self.num_workers = num_workers

        print(f"Using device: {self.device}")
        self.model = self._load_model()
        self.dataloader = self._load_data()

    def _load_model(self) -> torch.nn.Module:
        """load model

        Returns:
            torch.nn.Module: the zoe model
        """
        print("Loading ZoeDepth model from torch.hub...")
        model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True)
        return model

    def _load_data(self) -> DataLoader:
        """load data loader

        Returns:
            DataLoader: the dataloader for model
        """
        print(f"Loading dataset split: {self.split}")
        dataset = ZoeDepthDataset(self.split)
        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)
        return dataloader

    def evaluate(self) -> dict:
        """eval the model

        Returns:
            dict: results
        """
        print("Evaluating model...")
        results = evaluate_zoedepth_model(
            self.model, self.dataloader, self.device
        )
        return results


def main() -> None:
    """run and eval the model
    """
    evaluator = ZoeDepthEvaluator(split="val", batch_size=1, num_workers=4)
    results = evaluator.evaluate()

    print("\n--- Final Results ---")
    for metric in [
        "MAE", "MSE", "RMSE", "AbsRel", "Delta1",
        "Delta2", "Delta3", "Inference Time (s)"
    ]:
        value = results.get(metric)
        try:
            value_float = float(value)
            print(f"{metric}: {value_float:.4f}")
        except (ValueError, TypeError):
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
