# Sudoku digitisation CNN

Our model takes a sudoku and returns the digitised version of the sudoku as a 9x9 matrix. It uses a CNN on every individual cell in the sudoku in order to classify the number.

## Prerequisites

The dependencies can be seen listed in the requirements.txt file. The python version has to be **python 3.10** in order for the dependencies to work correctly.

## Launching the API

In order to run the API, run the following command line in your terminal:

- uvicorn fastAPI:app --reload

Once you see the line **Application startup complete**, the API is launched successfully. You can go to [this website](http://127.0.0.1:8000/docs) in order to preview the API using fastAPI.

From that website, click **try it now** and upload a png, jpg or jpeg of an already cropped sudoku.
You can find an example to try it on in the repo, the file called **sudoku.png**

Once you click **execute**, a matrix of the sudoku will be returned.