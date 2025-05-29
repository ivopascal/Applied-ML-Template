# Applied ML Project 🛠️

## Description of the project
Running a restaurant comes with high costs and complex logistics. Two major challenges are managing inventory to avoid waste and scheduling staff efficiently. While many restaurants keep track of reservations and actual guest counts, it's still tough to predict future demand accurately without the help of advanced tools. For this project, we will focus on solving a real-world problem in colaboration with "Weeva" restaurant, where one of our team members works. Our main goal is to build a model that helps predict how many guests the restaurant will have on a given day. If time allows, we’d also like to explore which menu items are most frequently ordered. We believe that factors like weather, the day of the week, and reservation counts are the key when making reliable predictions.


## Data Preprocessing
In this step of the project, we curated the dataset to predict the daily guest attendance at the restaurant and to rank the menu items by how often they're ordered. 

- **Guest data**: Records daily attendance of people at the restaurant.

- **Weather data**: Includes weather information in Groningen from 2018 to 2025.

- **Calendar data**: Tells what day of the week each date is (e.g., Monday, Tuesday, etc.).

- **School holiday data**: Shows whether each date is a Dutch school holiday or not.
  
- **Public holiday data**: Lists official holidays in the Netherlands and Germany.

- **Menu sales data**: Tracks how many times each dish was ordered each day (used to measure popularity).
  
### Steps we followed
1. We made sure all datasets use the same dates and removed outliers. We considered an outlier to be any date in the covid period (COVID_WINDOWS = [
    ("2020-03-01", "2020-05-31"),
    ("2020-12-01", "2021-06-30"),
    ("2021-11-01", "2022-01-31"),
]), and any entry with a number of guests that is not in the range (1, 400) (with 400 being an educated guess of restaurant's capacity).
2. We applied one-hot encoding for categorical variables.
3. We reorganized and reshaped time-series data.
4. We took all the menu items, and for each of them, we made a column in which we put their rank, ranging from most-ordered to least-ordered. 

## Splitting the data
### Steps we followed
1. We took the **last 365 days** of the dataset for validation and testing.
2. The rest was used for **training** (about 80% of the total data).
3. To make sure validation and test data are well-balanced, we assigned the **even-numbered days** to the **validation set** and the **odd-numbered days** to the **test set**. Since a week has an even number of days, the validation and test data will alternate in which days will contain.

## API
We created an API that allows users to send input and get a prediction back, from a trained model. The api offers the option to use and compare two models: random guesser as well as a linear regression model. It also includes proper input validation and returns clear responses, handling HTTPExceptions when something goes wrong.

### Structure
```
restaurant_guest_forecasting/
├── api/
│   ├── app.py              
│   ├── input.py           
│
├── models/
│   ├── random_guesser/
│   │   └── random_regression_guesser.py   
│   └── utils/
│       ├── evaluate_models.py  
│       ├── load_models.py       
│       └── train_models.py
|       └── saved_models 

```

- **app.py**: The main FastAPI application file.

- **input.py**: Defines the expected input data format using Pydantic.

- **random_regression_guesser.py**: Implements a Random Regression Guesser that always predicts the average value of the target in the training dataset.

- **evaluate_models.py**: Runs the given model on the validation data and returns the Mean Squared Error (MSE).
  
- **load_models.py**: Loads the saved models.

- **train_models.py**: trains a given model (either the Random Guesser or linear regression), and then saves it in *saved_models* directory.

### How to install dependencies and launch the API
1. Open a terminal
```bash
code here
```

2. Create a virtual environment
```bash
code here
```

3. Activate the virtual environment
```bash
code here
```

4. Install dependencies
```bash
code here
```

5. Launch the API
```bash
code here
```

6. Open the API in your own browser
```bash
code here
```

### Expected input format
```bash
code here
```

### Endpoints
- **POST /predict_guests/random**: Predict the number of guests using a random guesser (baseline model that always predicts the average guest count in the training set).
**Output**
```bash
code here
```

- **POST /predict_guests/model**: Predict the number of guests using a trained Linear Regression model.
**Output**
```bash
code here
```

- **GET /predict_guests/random/eval**: Returns the validation MSE for the random guesser.
**Output**
```bash
code here
```

- **GET /predict_guests/model/eval**: Returns the validation MSE for the linear regression model.
**Output**
```bash
code here
```

- **GET /predict_guests/compare**: Compare validation MSEs for both models.
**Output**
```bash
code here
```
