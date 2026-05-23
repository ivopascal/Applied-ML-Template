from torchvision import transforms


def get_vgg16_transform(image_size: int) -> transforms.Compose:
    """Create the image preprocessing transform for the VGG16 baseline"""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.449], std=[0.226]),
        ]
    )