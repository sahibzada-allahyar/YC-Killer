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


if __name__ == "__main__":
    app() 