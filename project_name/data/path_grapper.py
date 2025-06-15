import os
import threading

path_multithread_output: list[list[str]] = []


def grab_data_from_folder(main_folder: str) -> None:
    """Grabs the datapoint names from the major folder given

    Args:
        main_folder (str): the major folder liek 'indoor' or 'outdoor'
    """
    datapoint_directories: list[str] = [main_folder]
    for _ in range(3):
        new_datapoint_directories = []
        for path in datapoint_directories:
            sub_folders = os.listdir(path)
            for folder in sub_folders:
                new_datapoint_directories.append(os.path.join(path, folder))

        # get rid of all duplicates
        datapoint_directories = new_datapoint_directories

    # get rid of file extension and type description.
    for ending in [".png", "_depth.npy", "_depth_mask.npy"]:
        datapoint_directories = [_.removesuffix(ending)
                                 for _ in datapoint_directories]

    path_multithread_output.append(datapoint_directories)


def get_all_data_path_names(data_folder: str) -> list[list[str]]:
    """Gets all data point paths from train or val folders.

    Args:
        data_folder (str): the main folder to
        get the path names like train/val.

    Returns:
        list[list[str]]: two list that contain all data point path names.
    """
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  data_folder)

    datapoint_directories_main = []

    for folder in os.listdir(data_directory):
        datapoint_directories_main.append(os.path.join(data_directory,
                                                       folder))

    threads: list[threading.Thread] = []
    for main_folder in datapoint_directories_main:
        threads.append(threading.Thread(target=grab_data_from_folder,
                                        args=(main_folder, )))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    all_datapoint_folder_pathnames = path_multithread_output.copy()
    path_multithread_output.clear()

    return all_datapoint_folder_pathnames


def get_train_data_folders() -> list[str]:
    """Get the data folders path names in the subsetted data.

    Returns:
        list[str]: what subset to the get like 'train' or 'val'.
    """
    file_directory = __file__
    data_directory = os.path.join(os.path.split(file_directory)[0],
                                  "subset_data")

    train_data_folders = []

    for folder in os.listdir(data_directory):
        train_data_folders.append(os.path.join(data_directory, folder))

    return train_data_folders


def main() -> None:
    """
    To run stand alone functions
    """
    print(get_all_data_path_names("train"))


if __name__ == '__main__':
    main()
