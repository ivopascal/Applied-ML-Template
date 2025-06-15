import streamlit as st
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from os.path import split, join

from project_name.models.cnn import CNNBackbone
from project_name.models.Preprocessing_class import Preprocessing


def main() -> None:
    """
    Main function where the stream lit app runs.
    """
    if not ("device" in st.session_state):
        st.session_state["device"] = torch.device("cuda"
                                                  if torch.cuda.is_available()
                                                  else "cpu")

    if not ("model" in st.session_state):
        file_location = split(__file__)[0]
        st.session_state["model"] = CNNBackbone(pretrained=True).to(
            st.session_state["device"]
        )
        st.session_state["model"].load_state_dict(torch.load(
            join(file_location, "cnn_best.pth"), weights_only=True))
        st.session_state["model"].eval()

    if not ("preprocess" in st.session_state):
        st.session_state["preprocess"] = Preprocessing((200, 200))

    st.title("Streamlit demo for Applied Machine Learning: Depth prediction.")
    st.divider()
    intro_paragraph = """
    This is a stream lit demo for the AML project of group 16.
    The project is a depth estimating model from a RGB image.

    In this demo you will be allowed to upload an image covert it using our
    model and download the resulting depth image from it.
    """
    st.markdown(intro_paragraph)
    st.divider()
    upload_text = """
    Here you will upload an RGb 8-bit colour image for depth estimation.
    """
    st.markdown(upload_text)
    st.session_state["upload_image"] = st.file_uploader(label="Upload image",
                                                        type=[".png", ".jpeg"])

    if ("upload_image" in st.session_state and
            st.session_state["upload_image"] is not None):
        rgb_image = np.array(Image.open(st.session_state["upload_image"]))
        image_size = rgb_image.shape
        st.image(rgb_image)
        st.write(f"height: {image_size[0]}, width: {image_size[1]}, " +
                 f"channels: {image_size[2]}")
        st.session_state["rgb_image"] = rgb_image

        if image_size[2] > 3:
            st.error("We do not accept RGBa images.")

    st.divider()
    if (not ("upload_image" in st.session_state) or
            st.session_state["upload_image"] is None):
        if "depth_output" in st.session_state:
            st.session_state.pop("depth_output")
        return

    tile_size = st.number_input("tiling size of model",
                                1,
                                step=1,
                                value=200)
    st.write(f" the current tile size is {tile_size}")

    if st.session_state["preprocess"].tile_size[0] != tile_size:
        st.session_state["preprocess"].tile_size = (tile_size, tile_size)

    run_model = st.button("Start conversion.")

    if run_model and "rgb_image" in st.session_state:
        st.write("work in progress")

        model = st.session_state["model"]
        input_image = st.session_state["rgb_image"]
        pre_post_process = st.session_state["preprocess"]

        tiles = pre_post_process.tile_with_padding(input_image)

        tensor_input_image = torch.tensor(
            tiles,
            device=st.session_state["device"],
            dtype=torch.float).permute(0, 3, 1, 2)

        with torch.no_grad():
            tiles_output = model(tensor_input_image)

        depth_output = pre_post_process.reconstruct_depth(
            tiles_output.squeeze().cpu().numpy())

        st.session_state["depth_output"] = depth_output

    if "depth_output" in st.session_state:
        pre_post_process = st.session_state["preprocess"]
        image_output = pre_post_process.depth_to_rgb(
            st.session_state["depth_output"])

        st.image(image_output, clamp=True)

        depth_image_size = st.session_state["depth_output"].shape
        st.write(f"height: {depth_image_size[0]}," +
                 f" width: {depth_image_size[1]}, " +
                 "channels: 1")

    st.divider()
    if not ('depth_output' in st.session_state):
        return

    file_name = st.text_input("file name",
                              "depth_map")
    prepare_export = st.button("prepare depth map export")

    if file_name == "":
        st.error("empty name can not generate file")
        return

    if prepare_export and st.session_state["depth_output"].any():
        with BytesIO() as buffer:
            np.save(buffer, st.session_state["depth_output"])
            st.download_button("download depth map",
                               buffer,
                               file_name + ".npy")

    st.divider()


if __name__ == '__main__':
    main()
