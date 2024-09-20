# Online Machine Learning with SGD Regressor

## Overview
This project demonstrates how to perform online learning using the `SGDRegressor` (Stochastic Gradient Descent Regressor) from Scikit-learn. Online learning is useful when dealing with large datasets that cannot be processed all at once, as it allows the model to update incrementally with each new batch of data.

### Key Concepts:
- **Online Learning**: The model is updated incrementally using batches of data instead of training on the entire dataset at once.
- **SGD Regressor**: A linear regression model optimized using stochastic gradient descent.

## Code Explanation:
1. **Data Generation**: Random data is generated using `numpy`. 
    - `X`: A feature matrix with 1 sample and 500 features.
    - `y`: Random target values for the corresponding samples.
    
2. **Model Initialization**: The `SGDRegressor` from Scikit-learn is initialized for linear regression using stochastic gradient descent.

3. **Partial Fitting**: 
    - The model is updated incrementally with two batches of data. The `partial_fit` method is used to fit the model on each batch.
    - The time taken to fit the model on the first batch of data is measured using Python's `time` module.

4. **Online Update**: The model is fitted again on a new batch of data, simulating the incremental learning process typical of online learning models.

## How to Run the Code:
1. Ensure that Python and Scikit-learn are installed on your machine.
2. You can install Scikit-learn using:
    ```
    pip install scikit-learn
    ```
3. Run the script:
    ```
    python online_ml_sgd_regressor.ipynb
    ```

## Requirements:
- Python 3.x
- Scikit-learn
- Numpy

## Next Steps:
- You can modify the number of samples or features to see how the model performs with different dataset sizes.
- Try adding more batches of data to simulate an ongoing online learning scenario.