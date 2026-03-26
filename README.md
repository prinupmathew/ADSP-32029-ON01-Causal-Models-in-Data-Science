# causal-inference-in-python-code
Code for the Book [Causal Inference in Python](https://www.oreilly.com/library/view/causal-inference-in/9781098140243/)

## Prerequisites

1. `uv` - [Installation instruction](https://docs.astral.sh/uv/getting-started/installation/)
2. `git` - [Installation instruction](https://github.com/git-guides/install-git)

## Setting Up a Local Python Virtual Environment

1. Clone the repository
```
git clone https://github.com/jeongyoonlee/causal-inference-in-python-code.git
```

2. Create and activate virtual environment
```
cd causal-inference-in-python-code
uv venv
source .venv/bin/activate

or 

conda env create --file "causal-inference-program.yml"
conda env update --file "causal-inference-program.yml" --prune
```

3. Install dependencies
```
uv pip install .
```
