# API

## How to use
- cd into the /api folder 
- run: uvicorn api:app --reload
---

## Routes
- /prediction is the only route needed for this API
  - It is a POST route that accepts an image 
  - This image will be send to make a prediction with the ML model
  - When a prediction is given this will be returned to the request