
from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
import os
import sys
from torch import load


def load_model() -> XrayClassifierBase:
    path = sys.argv[1]
    if not os.path.exists(path):
        raise FileNotFoundError(f"path {path} not found")
    if "densenet" in path and "pretrained" in path:
        classifier = XrayClassifierBase("densenet", pretrained=True, model=load(path, weights_only=False))
    if "densenet" in path and "scratch" in path:
        classifier = XrayClassifierBase("densenet", pretrained=False, model=load(path, weights_only=False))
    if "vgg16" in path:
        classifier = XrayClassifierBase("vgg16", model=load(path, weights_only=False))
    return classifier



if __name__ == "__main__":
    model = load_model()
    model.evaluate()