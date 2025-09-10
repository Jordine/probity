# AI Debate System with Probe-Based Deception Detection

## Overview

This system implements debate-based AI oversight with optional integration of deception probes. The core idea is to have two AI models debate opposing positions while a judge evaluates their arguments. Probes trained to detect deception can be used to augment the debate in three ways:

1. **Baseline**: Standard debate without probes
2. **Debater Access**: Debaters can see probe scores on opponent's statements
3. **Full Access**: Both debaters and judge see all probe scores
4. **Judge Only**: Only the judge sees probe scores

## Installation

### Requirements

```bash
pip install torch transformers transformer-lens
pip install openai anthropic requests
pip install pandas numpy matplotlib seaborn
pip install datasets tqdm
```

### Environment Variables

Set API keys as environment variables (optional if providing via CLI):

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Quick Start

### 1. Train Probes (using existing probe training scripts)

```bash
# Train probes on your model first
python scripts/probe_training_ntml.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --probe_types logistic pca meandiff \
    --layers 15 \
    --probe_save_dir ./probes/llama_8b
```

### 2. Run a Baseline Debate (No Probes)

```bash
python -m debate.run_debate \
    --debate_type baseline \
    --honest_model meta-llama/Llama-3.1-8B-Instruct \
    --dishonest_model meta-llama/Llama-3.1-8B-Instruct \
    --judge_model gpt-4 \
    --honest_provider local \
    --dishonest_provider local \
    --judge_provider openai \
    --dataset apps \
    --n_problems 5 \
    --max_rounds 3
```

### 3. Run Debate with Probe Detection

```bash
python -m debate.run_debate \
    --debate_type full_access \
    --honest_model meta-llama/Llama-3.1-8B-Instruct \
    --dishonest_model meta-llama/Llama-3.1-8B-Instruct \
    --judge_model gpt-4 \
    --honest_provider local \
    --dishonest_provider local \
    --judge_provider openai \
    --probe_dir ./probes/llama_8b \
    --probe_types logistic pca \
    --probe_layer 15 \
    --dataset apps \
    --n_problems 5
```

### 4. Analyze Results

```bash
python -m debate.analyze_results ./debate_results/your_experiment_name
```

## Debate Types

### Baseline
- Standard debate without any probe information
- Tests pure argumentation ability

### Debater Access
- Debaters can see deception scores on opponent's statements
- Allows debaters to call out potentially deceptive claims
- Requires local models for both debaters

### Full Access
- Everyone (debaters and judge) sees all probe scores
- Maximum transparency about potential deception
- Requires local models for debaters

### Judge Only
- Only the judge sees probe scores
- Judge uses deception information for decision making
- Debaters argue without probe assistance

## Supported Models

### Local Models (Support Probes)
- LLaMA family: `meta-llama/Llama-3.1-8B-Instruct`, `meta-llama/Llama-3.3-70B-Instruct`
- Qwen family: `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`
- Any HuggingFace model compatible with TransformerLens

### API Models (Judge/Overseer)
- OpenAI: `gpt-4`, `gpt-4-turbo-preview`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`
- OpenRouter: Access to many models via unified API

## Dataset Support

### APPS Dataset
- Programming problems with solutions
- Automatic backdoor injection for testing
- Difficulty levels: introductory, interview, competition

### Custom Datasets
- Load from JSON file
- Format: List of problems with honest/backdoored solutions

## Configuration Options

### Core Arguments
- `--debate_type`: Type of debate (baseline, debater_access, full_access, judge_only)
- `--honest_model`: Model for honest debater
- `--dishonest_model`: Model for dishonest debater
- `--judge_model`: Model for judge
- `--max_rounds`: Number of debate rounds (default: 3)

### Probe Configuration
- `--probe_dir`: Directory containing trained probes
- `--probe_types`: Types of probes to use (logistic, pca, meandiff)
- `--probe_layer`: Layer to extract activations from
- `--probe_threshold`: Threshold for flagging deceptive claims (default: 0.7)

### Dataset Options
- `--dataset`: Dataset to use (apps, custom)
- `--n_problems`: Number of problems to evaluate
- `--difficulty`: APPS difficulty filter

## Output Structure

```
debate_results/
└── experiment_name/
    ├── config.json          # Experiment configuration
    ├── dataset.json         # Dataset used (for reproducibility)
    ├── results.csv          # Summary results
    ├── results.json         # Detailed results with probe scores
    └── analysis_plots.png   # Visualization plots
```

## Validation Requirements

The system automatically validates configurations:

1. **Probe Compatibility**: 
   - Probes can only run on local models
   - API models cannot have probes applied to them

2. **Debate Type Requirements**:
   - `debater_access`, `full_access`, `judge_only` require probe directory
   - These types require appropriate local models for probe application

## Examples

See `example_debate_runs.sh` for complete examples of different setups.

## Troubleshooting

### Common Issues

1. **"Debate type requires local model for probes"**
   - Solution: Use local models (not API) for debaters when using probe-based debate types

2. **"Probe directory not found"**
   - Solution: Train probes first using the probe training scripts

3. **"API key not found"**
   - Solution: Set environment variables or provide via `--*_api_key` arguments

4. **"Out of memory"**
   - Solution: Use smaller models or reduce batch size

## Analysis Features

The analysis script provides:
- Win rate statistics
- Probe score distributions by role
- Performance by problem difficulty
- Comparison across multiple experiments
- Visualization plots

## Future Extensions

The system is designed to be extensible for:
- Additional debate formats (multi-round, multi-agent)
- Different probe types and architectures
- Alternative datasets beyond APPS
- Custom evaluation metrics
- Real-time debate monitoring

## Citation

If you use this system in research, please cite:
- The probity probe training framework
- The relevant debate oversight papers
- The APPS dataset if used