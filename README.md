# Applied ML Project 🛠️

## Data Preprocessing
In this step of the project, we curated the dataset to predict the daily guest attendance at the restaurant and to rank the menu items by how often they're ordered.

- **Guest data**: Records daily attendance of people at the restaurant.

- **Weather data**: Includes weather information in Groningen from 2018 to 2025.

- **Calendar data**: Tells what day of the week each date is (e.g., Monday, Tuesday, etc.).

- **School holiday data**: Shows whether each date is a Dutch school holiday or not.
  
- **Public holiday data**: Lists official holidays in the Netherlands and Germany.

- **Menu sales data**: Tracks how many times each dish was ordered each day (used to measure popularity).
  
### Steps we followed
1. We made sure all datasets use the same dates and removed unwanted ones.
2. We applied one-hot encoding for categorical variables.
3. We reorganized and reshaped time-series data.
4. We took all the menu items, and for each of them, we made a column in which we put their rank, ranging from most-ordered to least-ordered. 

## Splitting the data
### Steps we followed
1. We took the **last 365 days** of the dataset for validation and testing.
2. The rest was used for **training** (about 80% of the total data).
3. To make sure validation and test data are well-balanced, we assigned the **even-numbered days** to the **validation set** and the **odd-numbered days** to the **test set**.

## API
We created an API that allows users to send input and get a prediction back. It also includes proper input validation and returns clear responses, handling HTTPExceptions when something goes wrong.

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
│       └── used_models.py       

```

- **app.py**: The main FastAPI application file.

- **input.py**: Defines the expected input data format using Pydantic.

- **random_regression_guesser.py**: Implements a Random Regression Guesser that always predicts the average value of the target in the training dataset.

- **evaluate_models.py**: Runs the given model on the validation data and returns the Mean Squared Error (MSE).
  
- **load_models.py**: Loads the saved models.

- **used_models.py**: trains a given model (either the Random Guesser or linear regression), and then saves it.

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
- **POST /predict_guests/random**: Predict the number of guests using a random guesser (baseline model that always predicts the average guest count).
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
