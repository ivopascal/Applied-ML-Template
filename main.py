import sys
from pathlib import Path
import argparse
from project_name.Training.model_trainer import train_cnn
from project_name.Training.Evaluation.evaluate import run_evaluation

project_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_dir))


def main() -> None:
    """Main function to run or validate model.
    """
    parser = argparse.ArgumentParser("Depth Estimation Pipeline")
    subs = parser.add_subparsers(dest="cmd", required=True)

    # train
    sub1 = subs.add_parser("train")
    sub1.add_argument("model", choices=["cnn"], help="Which model to train")
    sub1.add_argument("--epochs", "-e", type=int, default=20)
    sub1.add_argument("--batch-size", "-b", type=int, default=8)
    sub1.add_argument("--lr", type=float, default=1e-4)
    sub1.add_argument("--freeze", type=int, default=5)
    # python main.py train --epochs 20 --batch-size 8 --lr 1e-4 --freeze 5 cnn

    # evaluate
    sub2 = subs.add_parser("evaluate")
    sub2.add_argument("checkpoint", help="Path to .pth file")
    sub2.add_argument("--batch-size", "-b", type=int, default=8)
    # python main.py evaluate cnn_best.pth --batch-size 8

    args = parser.parse_args()

    if args.cmd == "train":
        train_cnn(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            freeze_epochs=args.freeze
        )
    else:
        run_evaluation(
            checkpoint=args.checkpoint,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
