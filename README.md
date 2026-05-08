# Tracking My Chess Career Through Blitz, Rapid, and Bullet

Tom White  
CPSC 222, Spring 2026

## Project Overview

This quantified-self project analyzes my personal Chess.com games to study which factors are most related to winning and losing. The project uses game-level records from Chess.com plus a generated calendar table so the data can be joined by date and summarized at a daily level.

The classification task predicts whether a decisive game is a win or a loss. Draws are kept for descriptive analysis but excluded from the supervised machine learning task.

## Repository Organization

- `quantified_self.ipynb`: technical report with narrative, visualizations, hypothesis tests, and classification results.
- `utils.py`: reusable loading, cleaning, feature engineering, plotting, statistics, and classifier helper functions.
- `imsogarb69_all_2604212016.csv`: raw personal Chess.com export.
- `games_clean.csv`: cleaned game-level table with engineered features.
- `calendar_table.csv`: daily calendar table joined to games by `date`.
- `daily_summary.csv`: one row per calendar day from the first game date through the last game date.
- `requirements.txt`: Python dependencies.

## Dataset Files

The raw data comes from my Chess.com account export and contains 1,678 rows before filtering. The cleaned dataset covers April 9, 2021 through September 6, 2025. It includes 1,623 blitz, rapid, and bullet games across 331 active playing days. The daily summary and calendar tables contain 1,612 calendar days, including zero-game days, which makes the daily sampling explicit.

Important attributes include time control, user color, user rating, opponent rating, rating difference, weekday, time-of-day bucket, opening family, recent win rate, and the class label `result_group`.

## How To Run

The project is intended to run in Jupyter Lab with the Anaconda Python Distribution v3.12.

```bash
pip install -r requirements.txt
jupyter lab
```

Open `quantified_self.ipynb` and run all cells from top to bottom. To regenerate the prepared CSV files from the raw export, run this in a Python session:

```python
import utils
utils.save_prepared_tables()
```

## Dependencies

The project uses pandas, NumPy, Matplotlib, Seaborn, SciPy, and scikit-learn.
