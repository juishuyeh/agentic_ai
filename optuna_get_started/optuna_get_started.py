import optuna
from finlab import data
from finlab.backtest import sim


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for Optuna optimization.
    Args:
        trial: Optuna trial object.
    Returns:
        Negative Sharpe ratio (since Optuna minimizes by default).
    """
    # Suggest parameter: moving average window between 5 and 60
    ma_window = trial.suggest_int("ma_window", 5, 60)

    # Get price data
    close = data.get("price:收盤價")
    sma = close.average(ma_window)

    # Simple crossover strategy
    entries = close > sma
    exits = close < sma

    position = entries.hold_until(exits, nstocks_limit=10, rank=None, take_profit=0.2)

    # Run backtest
    report = sim(position, upload=False)
    stats = report.get_stats()
    sharpe = stats["daily_sharpe"]

    # Optuna max, so return positive Sharpe
    return sharpe


def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("Best parameters:", study.best_params)
    print("Best Sharpe ratio:", study.best_value)


if __name__ == "__main__":
    main()
