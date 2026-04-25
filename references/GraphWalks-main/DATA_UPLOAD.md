# Data Upload Requirements for GraphWalks

## Overview
This environment requires the GraphWalks dataset to be uploaded to OpenReward cloud storage at `/orwd_data`.

## Dataset Information
- **Source**: HuggingFace `openai/graphwalks`
- **Size**: ~1,150 tasks (600 parents, 550 bfs)
- **Format**: Parquet file
- **File Size**: ~2-3 MB

## Directory Structure
```
/orwd_data/
└── data.parquet
```

## File Description
- **data.parquet**: Complete GraphWalks dataset containing:
  - `prompt`: Full task description with graph edges and operation
  - `answer_nodes`: List of correct node names
  - `problem_type`: Either "parents" or "bfs"
  - `prompt_chars`: Prompt length metric

## Local Development

For local testing, download the dataset to the GraphWalks directory:

```bash
cd /home/ross/Documents/or_envs/newenvs/GraphWalks
python -c "from datasets import load_dataset; import pandas as pd; ds = load_dataset('openai/graphwalks', split='train'); pd.DataFrame(ds).to_parquet('data.parquet')"
```

The environment will automatically detect and use the local `data.parquet` file when `/orwd_data/` is not available.

## Production Upload

1. Download the dataset locally using the command above
2. Go to https://openreward.ai and navigate to your namespace settings
3. Upload `data.parquet` to the storage for the `EnvCommons/GraphWalks` namespace
4. The file should be placed at `/orwd_data/data.parquet` in the deployed environment
5. The environment will automatically detect and use `/orwd_data/data.parquet` when deployed

## Important Notes

- **DO NOT** commit `data.parquet` to the git repository (it's in .gitignore)
- The dataset must be uploaded to cloud storage separately from the code deployment
- The Dockerfile intentionally does NOT include data.parquet to keep the image small
- The environment has fallback logic to load from HuggingFace if neither path has data (requires internet)
