# Prediction Intervals for Gradient Boosting Regression

This project follows closely the [example](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-quantile-py) provided by `Scikit-learn`!

The project shows how quantile regression can be used to create prediction intervals.

## Code structure

We start by generating some random input, following a function $f(x)=xsin(x)$, with a random noise term that follows a centered 
log-normal distribution. We make the noise a bit more interesting by making its amplitude depend on the input variable x.

We then fit gradient boosting models trained with the quantile loss and alpha=0.05, 0.5, 0.95.
The models obtained for alpha=0.05 and alpha=0.95 produce a 90% confidence interval.

## How to run it

Just run it with:
~~~
python prediction_interval.py
~~~
Make sure you have installed `numpy`, `matplotlib` and `scikit-learn` packages on your system! 
