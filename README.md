# VayuBench

A comprehensive benchmark and deployed system for evaluating Large Language Models on multi-dataset air quality analytics using real Indian environmental data.

## Overview

**VayuBench** is the first executable benchmark for air quality analytics, featuring 5,000 natural language queries paired with verified Python code across seven categories. **VayuChat** is the deployed interactive assistant that demonstrates real-world application of the benchmark.

🔗 **Try VayuChat Live**: [https://huggingface.co/spaces/SustainabilityLabIITGN/VayuChat](https://huggingface.co/spaces/SustainabilityLabIITGN/VayuChat)

##  Repository Structure

###  Datasets
```
preprocessed/           # Processed datasets in pickle format for fast Python loading
├── main_data.pkl      # CPCB air quality measurements (2017-2024)
├── states_data.pkl    # Indian state demographics and area data  
└── ncap_funding_data.pkl # NCAP funding allocations (2019-2022)

aqi_downloader.ipynb   # Jupyter notebook to download fresh CPCB data
questions.csv          # Complete benchmark with 10,034 questions + metadata
```

###  Scripts
```
batch_generation.py    # Batch query processing across multiple LLMs
eval_pipeline.py      # Main evaluation harness with sandboxed execution
code_eval_utils.py    # Core evaluation utilities (exec@1, pass@k calculations)
run.sh               # Automated execution script with nohup
```

##  Quick Start

1. **Download the datasets** using `aqi_downloader.ipynb` or use our preprocessed files
2. **Generate model outputs**: 
   ```bash
   python batch_generation.py --model_name qwen2.5-coder-14b --questions_file questions.csv
   ```
3. **Evaluate performance**:
   ```bash
   python eval_pipeline.py --model_outputs generated_code.json
   ```
4. **Run full pipeline**:
   ```bash
   chmod +x run.sh && ./run.sh
   ```

##  Benchmark Categories

| Category | Code | Description | Examples |
|----------|------|-------------|----------|
| **Spatial Aggregation** | SA | Geographic grouping across locations | "Which station in Delhi had highest PM2.5 in Jan 2022?" |
| **Temporal Trends** | TT | Time-series analysis and trends | "How did PM2.5 vary across 2021 in Lucknow?" |
| **Spatio-Temporal** | STA | Combined space-time analysis | "Which state had worst PM10 in summer 2022?" |
| **Population-Based** | PB | Population-weighted exposure analysis | "What % of population lives where PM2.5 exceeds WHO limits?" |
| **Area-Based** | AB | Geographic area-normalized queries | "Which state has fewest stations relative to area?" |
| **Funding-Related** | FQ | NCAP policy and funding analysis | "Which financial year had highest average funding?" |
| **Specific Patterns** | SP | Pattern detection over time windows | "How many days did Mumbai violate PM2.5 limits?" |

##  Key Results

| Model | Size | exec@1 | pass@1 | Error Rate |
|-------|------|--------|--------|-----------|
| **Qwen3-32B** | 32B | **0.98** | **0.78** | 0.01 |
| Qwen2.5-Coder-14B | 14B | 0.90 | 0.69 | 0.06 |
| GPT-OSS-20B | 20B | 0.88 | 0.56 | 0.12 |
| Llama3.2-1B | 1B | 0.04 | 0.00 | 0.97 |

##  Extended Repository Features

### Detailed Error Analysis
The appendix in our paper provides comprehensive error categorization with real examples:

- **Column Errors**: Incorrect dataset column references (most common)
- **Name Errors**: Undefined variables and missing imports  
- **Syntax Errors**: Python grammar violations
- **Semantic Errors**: Logically flawed but executable code

### Category-wise Code Examples
We provide detailed incorrect vs. correct code comparisons for each category, showing exactly where and why LLMs fail:

```python
# Example: Spatial Aggregation Error
# Model returns PM2.5 value instead of state name
model_output = 31.3582  # Wrong: numerical value
correct_output = "Haryana"  # Correct: state name
```

### System Prompt Templates
Complete prompt engineering templates used for consistent evaluation across all models, including schema definitions and constraint specifications.

### Comprehensive Evaluation Metrics
- **exec@1**: Syntactic correctness (code runs without errors)
- **pass@k**: Functional correctness (outputs match expected results)
- **Error rate**: Proportion of failed executions with detailed categorization

##  Applications

VayuBench enables researchers and practitioners to:
- Evaluate LLM performance on domain-specific analytics
- Develop better code-generation models for environmental data
- Build trustworthy AI systems for policy decision-making
- Access air quality insights through natural language queries

---
**Demo**: [VayuChat Live Application](https://huggingface.co/spaces/SustainabilityLabIITGN/VayuChat)  
