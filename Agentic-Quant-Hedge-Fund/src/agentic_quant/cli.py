"""
Basic CLI interface.
"""
from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def hello():
    """Say hello."""
    typer.echo("Hello from Agentic Quant!")


@app.command()
def version():
    """Show version information."""
    typer.echo("Agentic Quant v0.1.0")


@app.command()
def test_transforms():
    """Test mathematical transforms."""
    import numpy as np
    from .core.registry import get
    
    # Test basic arithmetic
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Test if add transform works
    add_fn = get("add")
    result = add_fn(data, data)
    typer.echo(f"Add transform test: {result}")


@app.command()
def idea():
    """Generate a new trading strategy idea."""
    from .agents.strategy_generator import generate
    path = generate()
    typer.echo(f"Generated idea saved to: {path}")


@app.command()
def test_backtest():
    """Test backtesting functionality."""
    import numpy as np
    import polars as pl
    from .backtest.engine import backtest
    
    # Create sample data for testing
    timestamps = pl.date_range(
        start=pl.date(2024, 1, 1),
        end=pl.date(2024, 1, 5),
        interval="1d"
    )
    
    sample_data = pl.DataFrame({
        "timestamp": timestamps,
        "ticker": ["BTCUSDT"] * len(timestamps),
        "alpha": [0.1, -0.2, 0.3, -0.1, 0.2],
        "close": [100.0, 102.0, 98.0, 105.0, 103.0]
    })
    
    result = backtest(sample_data)
    typer.echo(f"Backtest completed. Final PnL: {result['cum_pnl'][-1]:.2f}")


if __name__ == "__main__":
    app() 