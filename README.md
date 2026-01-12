# Soothcheck

A project for investigating lie detection and verification in large language models.

## Overview

Soothcheck is an ongoing research project focused on understanding how language models process and represent truthfulness. This repository contains our experimental code, datasets, and analysis tools.

*This project is under active development - detailed documentation and features will be added as we build them out.*

## Dependencies

This project builds upon the excellent [Probity](https://github.com/curt-tigges/probity) library by [Curt Tigges](https://github.com/curt-tigges), which provides tools for neural network probing and interpretability research.

## Requirements

```bash
# Python 3.7+
pip install torch>=2.0.0
pip install transformers>=4.30.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install transformer_lens==2.15.4
pip install scikit-learn>=1.3.0
pip install datasets>=2.12.0
pip install tqdm>=4.65.0
pip install tabulate>=0.9.0
pip install pandas
pip install seaborn
```

## Installation

```bash
git clone https://github.com/AviParrack/Soothcheck.git
cd Soothcheck
pip install -r requirements.txt  # (requirements.txt to be added)
```

## Documentation

**[Full Usage Guide](docs/USAGE.md)** - Complete documentation for:
- Dataset generation (NTML contrastive datasets)
- Statement bank expansion
- Probe training (sklearn, PyTorch, max aggregation)
- Probe evaluation on deception benchmarks
- Recommended settings and hyperparameters

## Project Structure

- `docs/USAGE.md` - Complete usage documentation
- `data/statement-banks/` - Curated datasets of true/false statements
- `data/NTML-datasets/` - Generated contrastive datasets for probe training
- `scripts/` - Training and evaluation scripts
- `probity/` - Core probity library (READ ONLY - see LLMagents.md)
- `LLMagents.md` - Instructions for LLM agents working on this project

## Dataset

Currently includes statement banks with:
- 3,968 true statements (`truth_bank.csv`)
- 3,968 false statements (`lie_bank.csv`)
- Sourced from 10 different domains (animals, companies, cities, etc.)

## Contributing

See `LLMagents.md` for development guidelines and instructions for AI agents working on this project.

## Citation

If you use this work, please cite both this project and the underlying Probity library:

**Soothcheck:**
```
@software{soothcheck,
  author = {Parrack, Avi}, {Nguyen, Jord}, {Attubato, Carlo}
  title = {Soothcheck: Lie Detection in Large Language Models},
  year = {2025},
  url = {https://github.com/AviParrack/Soothcheck}
}
```

**Probity (underlying library):**
```
@software{probity,
  author = {Tigges, Curt},
  title = {Probity: A Toolkit for Neural Network Probing},
  year = {2025},
  url = {https://github.com/curt-tigges/probity}
}
```

## License

*License to be determined*

---

*This project is in active development. Check back for updates as we build out the functionality and documentation.*