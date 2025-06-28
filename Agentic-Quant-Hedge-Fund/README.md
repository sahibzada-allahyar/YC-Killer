# Agentic Quantitative Trading by Singularity Research

## Fully autonomous Hedge Fund, run end-to-end entirely by AI agents.

> 🌟 **Join Our Mission**: We're systematically democratizing quantitative finance by open-sourcing enterprise-grade trading systems that rival the most sophisticated hedge funds. If you have a stellar academic background and are excited about making advanced quant strategies accessible to everyone, please star this repository and [apply to join our team](https://www.singularityresearchlabs.com).



A complete, production-ready agentic quantitative trading system that brings institutional-grade algorithmic trading capabilities to everyone. Our mission is to democratize quantitative finance and break down the barriers that have kept sophisticated trading strategies locked behind the walls of elite hedge funds.

## Our Mission

We believe access to advanced quantitative trading strategies is a human right in the modern financial system. While hedge funds charge 2% management fees and 20% performance fees for strategies built on decades-old infrastructure, we're creating cutting-edge AI-powered trading systems and making them freely available to everyone. Our goal is to accelerate financial democratization and give retail traders the same tools used by institutional giants.

## Our Team

Our team consists of the most brilliant quantitative researchers and AI engineers in the world, including alumni from:
- Google DeepMind
- Harvard University
- MIT
- Stanford University
- Cambridge University
- Anthropic
- Citadel

## Features

### 🤖 **Agentic Alpha Generation**
- **Strategy Ideation Agent**: Generates market-neutral trading ideas based on behavioral finance principles
- **Alpha Engineering Agent**: Converts high-level ideas into executable mathematical expressions
- **Automated Feature Engineering**: Compiles trading signals using NumPy+Numba for maximum performance
- **Expression Parser**: Validates and compiles prefix-notation trading formulas
- **Strategy Backtesting**: Mean-variance neutral backtesting with realistic transaction costs

### 📊 **AI-Powered Strategy Generation**
- LLM-based idea generation rooted in market microstructure theory
- Automated conversion of trading concepts into mathematical expressions
- Support for complex multi-factor models and cross-sectional strategies
- Integration with open-source LLMs (Ollama, vLLM) for complete independence

### ⚡ **High-Performance Data Engine**
- Polars-based data processing for lightning-fast computations (1M+ rows/second)
- Numba JIT compilation for critical path optimizations
- Support for tick-level cryptocurrency data via Tardis
- Automated OHLC aggregation across multiple timeframes
- Memory-efficient streaming data pipeline

### 🎯 **Enterprise-Grade Backtesting**
- Dollar-neutral long/short equity strategies
- Realistic bid-ask spread modeling
- Position sizing with risk management
- Performance attribution and factor analysis
- Vectorized computation for rapid iteration

### 🐳 **Production-Ready Infrastructure**
- Docker containerization for easy deployment
- DVC for data version control and reproducibility
- Comprehensive CLI for end-to-end workflow management
- Extensible plugin architecture for custom transforms
- Complete test coverage and CI/CD integration

## Technical Architecture

Built with cutting-edge open-source technologies:

### **Core Stack**
- **Python 3.11+** with modern type hints and async support
- **Polars** for blazing-fast dataframe operations
- **NumPy + Numba** for high-performance numerical computing
- **Lark** for robust expression parsing with formal grammar
- **Typer** for intuitive command-line interface

### **AI/ML Components**
- **Ollama** for local LLM inference without external dependencies
- **Jinja2** for sophisticated prompt templating
- **Custom expression engine** with 20+ mathematical transforms
- **Automated feature hashing** for efficient caching and deduplication

### **Data Pipeline**
- **Tardis** integration for institutional-grade cryptocurrency data
- **Parquet** columnar storage for optimal I/O performance
- **Time-series resampling** across 7 standard frequencies (1min to 1day)
- **Streaming data processing** with memory-efficient lazy evaluation

## Project Structure

```
YC-Killer/
└── Agentic-Quant-Hedge-Fund/
    ├── src/
    │   └── agentic_quant/
    │       ├── agents/              # AI-powered strategy generators
    │       │   ├── feature_generator.py    # Alpha expression generator
    │       │   ├── llm_client.py          # LLM integration
    │       │   ├── prompts.py             # Prompt engineering
    │       │   └── strategy_generator.py   # Trading idea generator
    │       ├── backtest/            # Backtesting engine
    │       │   ├── cost_model.py          # Transaction cost modeling
    │       │   ├── engine.py              # Core backtesting logic
    │       │   ├── metrics.py             # Performance analytics
    │       │   └── report.py              # Report generation
    │       ├── core/                # Mathematical engine
    │       │   ├── datatypes.py           # Market data schemas
    │       │   ├── parser.py              # Expression parsing
    │       │   ├── registry.py            # Transform registry
    │       │   └── transforms.py          # Mathematical transforms
    │       ├── data/                # Data processing
    │       │   ├── features.py            # Feature computation
    │       │   ├── loader.py              # Data loading utilities
    │       │   └── ohlc_builder.py        # OHLC aggregation
    │       ├── orchestration/       # Pipeline management
    │       │   ├── pipeline.py            # Workflow orchestration
    │       │   └── scheduler.py           # Task scheduling
    │       ├── tests/               # Test suite
    │       │   ├── integration/           # Integration tests
    │       │   └── unit/                  # Unit tests
    │       └── cli.py               # Command-line interface
    ├── docker/                      # Containerization
    │   ├── Dockerfile
    │   └── compose.yaml
    ├── data/                        # Data directory structure
    │   ├── raw/                     # Raw market data (never committed)
    │   ├── processed/               # OHLC parquet files
    │   └── features/                # Computed alpha features
    ├── artifacts/                   # Generated artifacts
    │   ├── ideas/                   # Strategy ideas JSON
    │   └── alphas/                  # Alpha expressions JSON
    └── pyproject.toml              # Project configuration
```

## Prerequisites

- Python 3.11+
- Ollama (for LLM inference)
- Poetry or pip for dependency management
- Docker (optional, for containerized deployment)

## Installation

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Quant-Hedge-Fund
   ```

2. Install dependencies:
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -e .
   ```

3. Set up your LLM service:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh
   
   # Pull a model (e.g., Llama 3)
   ollama pull llama3
   ```

4. Configure environment (optional):
   ```bash
   cp .env.example .env
   # Edit .env to customize LLM endpoints
   ```

## Usage

### CLI Usage

1. **Generate a trading idea**:
   ```bash
   python -m agentic_quant.cli idea
   # Output: Saved → artifacts/ideas/idea_a1b2c3d4.json
   ```

2. **Convert idea to alpha expressions**:
   ```bash
   python -m agentic_quant.cli alphas artifacts/ideas/idea_a1b2c3d4.json --n 5
   # Output: Saved → artifacts/alphas/alpha_idea_a1b2c3d4_5.json
   ```

3. **Build features for a symbol**:
   ```bash
   python -m agentic_quant.cli build artifacts/alphas/alpha_idea_a1b2c3d4_5.json BTCUSDT
   # Output: Feature hashes with computed parquet files
   ```

4. **Run backtest**:
   ```bash
   python -m agentic_quant.cli run-backtest <feature_hash> BTCUSDT 1h
   # Output: Backtest results with performance metrics
   ```

### Docker Usage

1. Build the container:
   ```bash
   docker build -f docker/Dockerfile -t agentic-quant .
   ```

2. Run with docker-compose:
   ```bash
   docker-compose -f docker/compose.yaml up
   ```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_BASE_URL | Ollama API endpoint | http://localhost:11434 |
| OLLAMA_MODEL | LLM model to use | llama3 |
| CONTEXT_SIZE | Maximum context size | 32000 |
| PARALLEL_EXECUTION_LIMIT | Concurrent operations | 4 |
| DEBUG | Enable debug logging | false |

### Data Pipeline Configuration

Expected directory structure:
```
data/
├── raw/                    # Never committed to git
│   └── binance/
│       └── futures/
│           ├── quotes/     # Tardis quote data
│           └── trades/     # Tardis trade data
├── processed/              # OHLC parquet files
│   └── {symbol}/
│       └── ohlc/
│           └── {freq}/
└── features/               # Computed alpha features
    └── {hash}/
```

## Example Alpha Expression

The system generates expressions like:
```
ts_zscore(div(sub(high, low), close), 30)
```

Which computes: "30-period z-score of the high-low range normalized by close price" - a classic volatility-based mean reversion signal.

## Performance Benchmarks

- **Data Processing**: 1M+ rows/second with Polars
- **Expression Evaluation**: 10K+ expressions/second with Numba
- **Backtesting**: Full universe simulation in minutes, not hours
- **Memory Efficiency**: Streaming pipeline handles datasets larger than RAM

## Development

### Scripts

- Run tests:
  ```bash
  python -m pytest src/agentic_quant/tests/
  ```

- Type checking:
  ```bash
  mypy src/agentic_quant/
  ```

- Format code:
  ```bash
  black src/agentic_quant/
  ```

### Architecture

The system operates through several key components:

1. **Strategy Generation**: AI agents generate trading ideas and convert them to mathematical expressions
2. **Expression Engine**: Parses and compiles mathematical formulas using Lark grammar
3. **Data Pipeline**: High-performance data loading and feature computation
4. **Backtesting Engine**: Mean-variance neutral backtesting with realistic costs
5. **CLI Interface**: Comprehensive command-line tools for workflow management

## API Documentation

### Core Functions

#### Strategy Generation
```python
from agentic_quant.agents.strategy_generator import generate

# Generate a new trading idea
idea_path = generate()
```

#### Feature Computation
```python
from agentic_quant.data.features import FeatureSet

# Create and build features
fs = FeatureSet.from_expr("1h", "ts_zscore(close, 30)")
feature_file = fs.build("BTCUSDT", data_root, feature_root)
```

#### Backtesting
```python
from agentic_quant.backtest.engine import backtest

# Run backtest on feature data
results = backtest(df, lookahead=1)
```

## Research Papers & References

This system implements concepts from:
- **"The Man Who Solved the Market"** - Jim Simons and Renaissance Technologies
- **"Advances in Financial Machine Learning"** - Marcos López de Prado
- **"Machine Learning for Asset Managers"** - Marcos López de Prado
- **Academic literature** on market microstructure, behavioral finance, and systematic trading

## Apply to Collaborate

We're looking for exceptional quantitative researchers and AI engineers who share our vision of democratizing finance:

- **Quantitative Researchers**: PhD in Mathematics, Physics, Statistics, or Finance with experience in systematic trading
- **AI Engineers**: Background in large language models, agentic systems, or financial AI
- **Data Engineers**: Experience with high-frequency data processing and real-time systems
- **Open Source Enthusiasts**: Passionate about making sophisticated tools accessible to everyone

**Apply here**: [https://www.singularityresearchlabs.com](https://www.singularityresearchlabs.com)

## Contributing

We welcome contributions from the community:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-transform`)
3. Add comprehensive tests for new functionality
4. Ensure all linting and type checking passes
5. Submit a pull request with detailed description

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Please consult with a qualified financial advisor before making any investment decisions.

## Contact

- **Email**: [sahibzada@singularityresearchlabs.com](mailto:sahibzada@singularityresearchlabs.com)
- **Website**: [https://www.singularityresearchlabs.com](https://www.singularityresearchlabs.com)

## About Singularity Research

Singularity Research is dedicated to democratizing advanced technologies through open source initiatives. We believe in making institutional-grade AI and quantitative finance tools accessible to everyone, fostering innovation and breaking down barriers in finance and technology. Our goal is to accelerate technologies that empower individuals and challenge the monopolies of overfunded, underperforming institutions.

---

**Made with ❤️ by Singularity Research Labs**

*"The best way to predict the future is to create it, and the best way to create it is to make it open source."*
