from chest_xray.models.xraymodelbaseclass import XrayClassifierBase
import sys

def main():
    match sys.argv[1]:
        case "baseline":
            classifier = XrayClassifierBase(type="vgg16")
        case "pretrained":
            classifier = XrayClassifierBase(type="densenet", pretrained=True)
        case "scratch":
            classifier = XrayClassifierBase(type="densenet", pretrained=False)
        case _:
            raise Exception(f"{sys.argv[1]} is not one of the models, use baseline, pretrained, or scratch")
    classifier.trainModel()

if __name__ == "__main__":
    main()
