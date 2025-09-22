This repository contains the code, data processing pipeline, and experimental results for my tech stock forecasting project.

The study develops a two-tier optimization framework that integrates Genetic Algorithms (GA), Bayesian Optimization (BO), and NSGA-II for forecasting and trading strategy design on U.S. technology stocks (the “Magnificent Seven”).

📌 Project Overview

The project evaluates two model families:

	•	ARIMA–GARCH: a stable, interpretable econometric benchmark with strong downside risk control.
	•	LSTM: a deep learning alternative that captures nonlinear dependencies but is more volatile.

A hybrid optimization pipeline is proposed:

	1.	Tier 1 – Forecast-Oriented Optimization:

	•	GA for global hyperparameter search.
	•	BO for local refinement to minimize RMSE.

	2.	Tier 2 – Trading-Oriented Optimization:
 
	•	NSGA-II jointly optimizes Sharpe Ratio and Maximum Drawdown.
	•	Knee-point selection identifies robust, deployable strategies.

Backtesting utilizes walk-forward validation (2010–2020) with retraining intervals of 10, 20, and 42 days, incorporating transaction costs and evaluating metrics such as RMSE, Sharpe Ratio, Maximum Drawdown, Turnover, and Cumulative Return.
 
🛠️ Requirements

	•	Python 3.12
 
	•	PySpark, Pandas, NumPy
 
	•	TensorFlow/Keras
 
	•	Statsmodels, Arch
 
	•	Skopt, Pymoo
 
	•	Matplotlib

See requirements.txt for details.
