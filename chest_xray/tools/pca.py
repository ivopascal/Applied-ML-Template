import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from read_lists import get_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pickle

READ_EVERY: int = 100

def get_image_path():
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.normpath(os.path.join(script_path, "../../data/images/"))
    return file_path

def get_flattened_images(imgs):
    image_path = get_image_path()
    for i, img_name in enumerate(imgs["img_name"]):
        print(f"processing image {i}: {img_name}")
        full_path = os.path.join(image_path, img_name)
        img = Image.open(full_path).convert("L")
        img_resized = img.resize((64,64))
        img_array = np.array(img_resized) / 255
        yield img_array.flatten()

def tsne(images, perplexity = 30):
    scaled_data = StandardScaler().fit_transform(images)
    pca_pre = PCA(n_components = 50)
    prepped_data = pca_pre.fit_transform(scaled_data)
    t_sne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=200.0,
        init='pca',
        random_state=42
    )
    return t_sne.fit_transform(prepped_data)


def main():
    pickle_file = '/scratch/s3668320/AML/tsne_results.pkl'
    tsne_result = None    
    imgs = get_data()[['img_name', 'diseases']].copy().iloc[::READ_EVERY].copy()
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            print("pickle found")
            tsne_results = pickle.load(f)
    else:
        print("getting data")
        x = np.array([img for img in get_flattened_images(imgs)])
        print("doing tsne")
        tsne_results = tsne(x)
    print(tsne_results)
    with open('/scratch/s3668320/AML/tsne_results.pkl', 'wb') as f:
        pickle.dump(tsne_results, f)
    df_plot = pd.DataFrame({
        'tsne-1': tsne_results[:, 0].copy(),
        'tsne-2': tsne_results[:, 1].copy(),
        'disease': imgs["diseases"].apply(lambda x: x.split('|')[0])
    })
    plt.figure()
    sns.scatterplot(
        x='tsne-1',
        y='tsne-2',
        hue = 'disease',
        palette=sns.color_palette("hls", len(df_plot['disease'].unique())),
        data=df_plot,
        legend="full",
        alpha=0.7
    )
    plt.title("t-SNE visualization of NIH Chest X-ray Dataset")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("/scratch/s3668320/AML/data/proposal/tsne.png", bbox_inches="tight")

if __name__ == "__main__":
    main()
