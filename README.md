# Applied ML Final Project 🛠️

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
### Structure
