# Anonymous Code Submission

## Overview

This repository contains the implementation of five memory mechanisms for large language models (LLMs):

- **Baselines**: A-mem, mem0, MemoryBank, MemoryOS
- **Our Method**: OptiMem (an optimized version based on MemoryOS)

All methods are evaluated on the LoCoMo benchmark for long-term conversation memory.

## Repository Structure

```
.
├── A-mem/
├── mem0/
├── MemoryBank/
├── MemoryOS/
├── OptiMem/
├── requirements.txt
├── requirements_memorybank.txt
└── README.md
```

## Dataset

All methods use the LoCoMo dataset. The dataset file `locomo.json` (or `locomo10.json`) is placed inside each method's subfolder.

## Environment Setup

Two separate conda environments are required.

**Environment 1 for A-mem, mem0, MemoryOS, OptiMem:**

```bash
conda create -n llm_memory_base python=3.9
conda activate llm_memory_base
pip install -r requirements.txt
```

**Environment 2 for MemoryBank:**

```bash
conda create -n memorybank python=3.9
conda activate memorybank
pip install -r requirements_memorybank.txt
```

## Configuration

Before running any experiments, please configure your LLM API credentials. Each method uses OpenAI-compatible API calls. You need to set the following environment variables or fill in the corresponding configuration files within each subfolder:

- `API_KEY`: Your LLM service API key
- `BASE_URL`: The API endpoint URL (e.g., `https://api.openai.com/v1` or your custom endpoint)

Make sure these fields are properly configured in the code files before execution.


## Running Experiments

### 1. A-mem (Baseline)

Environment: `llm_memory_base`

```bash
cd A-mem
python eval_locomo.py --dataset data/locomo10.json --model deepseek-chat --backend openai --output A_mem_result.json
```

Control samples: Modify `allow_sample` in `eval_locomo.py`

Evaluation:

```bash
python grade_locomo.py
```

### 2. mem0 (Baseline)

Environment: `llm_memory_base`

```bash
cd mem0
python run_experiments.py --technique_type mem0 --method add
python run_experiments.py --technique_type mem0 --method search --output_folder results/ --top_k 30
```

Control samples: Modify sample indices in `pipeline_search.py` and `pipeline_add.py`

Evaluation:

```bash
python grade_locomo.py
```

### 3. MemoryBank (Baseline)

Environment: `memorybank`

```bash
cd MemoryBank
python forget_eval_locomo.py
```

Control samples: Modify parameters in `forget_eval_locomo.py`

Evaluation:

```bash
python grade_locomo.py
```

### 4. MemoryOS (Baseline)

Environment: `llm_memory_base`

```bash
cd MemoryOS
python Demo_5_locomo.py
```

Control samples: Modify model and sample index in `Demo_5_locomo.py`

Evaluation:

```bash
python demo_5_grade_locomo.py
```

### 5. OptiMem (Our Method)

Environment: `llm_memory_base`

```bash
cd OptiMem
python Demo_5_locomo.py
```

Control samples: Modify model and sample index in `Demo_5_locomo.py`

Evaluation:

```bash
python demo_5_grade_locomo.py
```

> **Note**: OptiMem follows the same workflow as MemoryOS but with optimized memory mechanisms.

## Output Files

Each method generates a JSON result file. Evaluation results are produced by the respective `grade_locomo.py` scripts.

## Dependencies

- `requirements.txt`: A-mem, mem0, MemoryOS, OptiMem
- `requirements_memorybank.txt`: MemoryBank

## Important Notes

- Each subfolder contains its own copy of the dataset file
- Results vary based on the sample indices processed

## Citation

[Citation information to be added upon paper publication]

## License

[License information to be added]