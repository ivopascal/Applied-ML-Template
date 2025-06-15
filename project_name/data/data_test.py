import matplotlib.pyplot as plt
from numpy import array, ndarray, load, concatenate
from scipy.stats import normaltest


multithread_data_test_output: list[tuple[str, ndarray]] = []


def test_dataset_normality(data: list[str], name: str) -> None:
    """Gets the data from given list of data point and plots a histogram and
    normality on them.

    Args:
        data (list[str]): list of path names to data points
        name (str): what name to give the plot
    """
    whole_data = array([])
    max_n = 0

    for data_point in data:
        matrix: ndarray = load(data_point + "_depth.npy")
        max_n = max(max_n, matrix.max())
        whole_data = concatenate([whole_data,
                                  matrix.flatten()[whole_data < 100]])

    normality = normaltest(whole_data)

    plt.hist(whole_data)
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title(f"{name} | norm: stat:{normality.statistic:0.2f}, " +
              f"p:{normality.pvalue:0.2f}")
    plt.show()


def threaded_make_data_array(data: list[str], name: str) -> None:
    """Get the data from list of path names. This is for threaded work.

    Args:
        data (list[str]): list of path names
        name (str): what name to give the plot that follows.
    """
    whole_data = array([])
    max_n = 0

    for data_point in data:
        matrix: ndarray = load(data_point + "_depth.npy")
        max_n = max(max_n, matrix.max())
        whole_data = concatenate([whole_data,
                                  matrix.flatten()[whole_data < 100]])

    multithread_data_test_output.append((name, whole_data))


def test_data(data: ndarray, name: str) -> None:
    """test normality and plot the histogram

    Args:
        data (ndarray): numpy array of 1 dimension of depth
        name (str): name of the data
    """
    normality = normaltest(data)

    plt.hist(data)
    plt.xlabel("Depth")
    plt.ylabel("Frequency")
    plt.title(f"{name} | norm: stat:{normality.statistic:0.2f}, " +
              f"p:{normality.pvalue:0.2f}")
    plt.show()
