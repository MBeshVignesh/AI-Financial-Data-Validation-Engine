# AI Financial Data Validation Engine (Hierarchical and Row-Level)

Built a deterministic validation engine that operates on structured hierarchy and measure data. For hierarchies, it validates structural integrity like parent-child relationships, level consistency, and leaf-node correctness. For numeric data, it validates both row-level matches and aggregated rollups. The key part is combining both — so it can detect not just value mismatches, but whether they’re caused by hierarchy issues.

The system has three layers: ingestion, validation, and reasoning. Validation is fully deterministic to ensure correctness, while AI is used before validation to interpret business context and after validation to explain discrepancies.

## Repo Tree

```text
.
├── .env.example
├── README.md
├── docs/
│   ├── architecture.md
│   └── images/
├── pyproject.toml
├── requirements.txt
├── scripts/
│   └── run_eval.py
├── src/
│   └── hierarchy_migration_validation_agent/
│       ├── agent/
│       ├── api/
│       ├── frontend/
│       ├── ingestion/
│       ├── normalization/
│       ├── rag/
│       ├── reporting/
│       ├── schemas/
│       ├── storage/
│       ├── utils/
│       └── validation/
└── tests/
```

## Features

- Upload-only workflow for one source workbook plus one target workbook.
- Ingestion support for either separate workbooks or a single multi-tab Excel workbook that contains Smart View sheets, mappings, and rules together.
- Ingestion support for flattened hierarchy tabs such as `Level 1 / Level 2 / Level 3 / Level 4`, path-style HCM tabs like `Entity / Business Unit / Department / Cost Center`, parent-child sheets, and generic ordered hierarchy columns such as `Company / Division / Team`.
- Deterministic validation engine for:
  - missing members in target
  - parent existence
  - source parent vs target parent mismatch
  - duplicate members
  - leaf/non-leaf consistency
  - level consistency
  - row-level hierarchy match
  - numeric value match
  - mapping completeness
  - optional rollup preservation for account totals
- Chroma RAG over mappings, validation rules, transformation notes, prior exceptions, and parsed hierarchy context.
- Strict local embeddings with `nomic-ai/nomic-embed-text-v1.5`.
- FastAPI backend and Streamlit demo UI built on the same workflow layer.
- JSON plus markdown reporting with likely causes and recommended actions.
- SQLite run registry and persistent Chroma index.

## Setup

1. Create and activate a virtual environment.
2. Install the package and dev dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Or, if you prefer a `requirements.txt` flow:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The `requirements.txt` flow now installs both the dependencies and the local package in editable mode, so the API, demo script, and Streamlit app can be run directly from the repo root.

3. Optional: start Ollama locally if you want LLM-generated explanations.

```bash
ollama run llama3.2
```

4. The app uses the local embedding model `nomic-ai/nomic-embed-text-v1.5` for RAG.
   If the model is not cached yet, it will be downloaded on first use unless you set `EMBEDDING_LOCAL_FILES_ONLY=true`.
5. Start the app you want:

```bash
python -m uvicorn hierarchy_migration_validation_agent.api.main:app --reload
```

```bash
python -m streamlit run src/hierarchy_migration_validation_agent/frontend/streamlit_app.py
```

## Demo Flow

1. Upload a source workbook and a target workbook in the Streamlit UI, or send them to `POST /ingest-excel`.
2. Click `Ingest Uploaded Excel`.
3. The app parses the uploaded workbooks directly in memory, detects hierarchy, measure, mapping, and rule sheets, and automatically rebuilds the RAG index from the uploaded source, target, mappings, rules, notes, and prior exceptions.
4. Click `Run Validation`.
5. Review the validation summary, row-level failures, check-by-check results, and downloadable JSON/markdown reports.
6. After the report is shown, the app clears uploaded source files, target files, and supporting docs so the next run starts clean. Chroma stays on disk and the next ingest uses a new upload-scoped collection.

## Demo Screenshots

### Source And Target Preview

<img width="2938" height="1650" alt="image" src="https://github.com/user-attachments/assets/7b29c7c4-90e5-43a9-821e-03eda9b61e8f" />



### Validation Checks And Row-Level Failures

<img width="2940" height="1638" alt="image" src="https://github.com/user-attachments/assets/44d12878-016c-44af-9df5-d881e2ce2ae0" />


### Start the API

```bash
python -m uvicorn hierarchy_migration_validation_agent.api.main:app --reload
```

### Start the Demo UI

```bash
python -m streamlit run src/hierarchy_migration_validation_agent/frontend/streamlit_app.py
```

### Run the 5-Case Evaluation

```bash
python scripts/run_eval.py
```

Useful options:

```bash
python scripts/run_eval.py --judge auto
python scripts/run_eval.py --judge ollama
python scripts/run_eval.py --judge rubric
python scripts/run_eval.py --format json
```

The evaluation script runs five benchmark scenarios, scores each case out of 10, aggregates the result to a percentage, and writes JSON plus markdown evaluation reports under `data/reports/evaluations/`.

Notes:

- `--judge auto` uses Ollama if available, otherwise falls back to a deterministic rubric.
- `--judge ollama` attempts to use the local Ollama model as the judge.
- `--judge rubric` skips the LLM judge and uses a deterministic scoring rubric.
- The evaluation harness uses a deterministic dummy embedding internally so the benchmark stays reproducible and fast.

## API Endpoints

- `POST /ingest-excel`
- `POST /build-index`
- `POST /validate`
- `GET /validation-report/{run_id}`
- `GET /health`

### Example Request

```bash
curl -X POST http://localhost:8000/ingest-excel \
  -F "source_files=@/path/to/source_workbook.xlsx" \
  -F "target_files=@/path/to/target_workbook.xlsx"
```

### Example Response

```json
{
  "ingested_at": "2026-03-18T16:45:00Z",
  "normalized_files": [
    {"name": "entity_source", "path": "in-memory://entity_source", "row_count": 16},
    {"name": "entity_target", "path": "in-memory://entity_target", "row_count": 16}
  ],
  "source_files": ["/path/to/source_workbook.xlsx"],
  "target_files": ["/path/to/target_workbook.xlsx"],
  "rag_document_count": 42,
  "warnings": []
}
```

Then run validation:

```bash
curl -X POST http://localhost:8000/validate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Validate the uploaded source hierarchy workbook against the uploaded target hierarchy workbook"
  }'
```

## Key Design Decisions

- `pandas` first: the MVP uses dataframe-based logic because the required validations are transparent, easy to test, and straightforward to port to SQL or Spark later.
- Direct workbook parsing: uploaded source and target files are parsed in memory instead of relying on persisted normalized CSVs just to run validation.
- SQLite for run state: report metadata is persisted locally without introducing external infrastructure.
- Chroma plus local Nomic embeddings: retrieval stays local while producing stronger semantic matches for mappings, rules, notes, and prior exceptions.
- Strict local embedding path: if the local Nomic model or persistent Chroma setup is unavailable, the app now fails loudly instead of falling back to weaker embeddings.
- Ollama as an optional reasoning layer: validation outcomes are deterministic, while summarization and explanation can use `llama3.2` when available.
- Shared services for API and UI: FastAPI and Streamlit call the same workflow layer so behavior stays aligned.

## Known Limitations

- The target zone is simulated locally and does not connect to real ADLS storage.
- The RAG layer requires the local Nomic embedding stack to load successfully. If `sentence-transformers`, `einops`, model files, or persistent Chroma storage are not available, indexing will fail and should be fixed before validation.
- The prompt box in the Streamlit UI is currently a dimension/retrieval hint, not a full natural-language rule planner. It can influence dimension selection and explanation wording, but it does not invent new validation rules.
- Streamlit and pytest are declared dependencies but may need installation in your local environment before running the UI or tests.
- Rollup preservation is currently focused on account totals rather than numeric balance reconciliation.
- The MVP assumes a reasonably consistent extract layout; broader workbook variability would need additional schema detection rules.
- Screenshot files are referenced in `docs/images/` and need to be added there for README image rendering.

## Testing

```bash
pytest
```

The automated suite currently covers embeddings, ingestion, normalization helpers, validation rules, RAG behavior, validation text generation, and end-to-end workflow coverage.

## Multi-Tab Workbook Support

`POST /ingest-excel` and the shared ingestion service can now handle:

- separate files like `Account_Hierarchy_Source.xlsx` and `Hierarchy_Mapping.xlsx`
- one combined workbook with multiple tabs for account Smart View data, entity Smart View data, mappings, and validation rules
- one source workbook plus one target workbook uploaded separately
- path-style target and source tabs such as `Level 1 / Level 2 / Level 3 / Level 4` or `Entity / Business Unit / Department / Cost Center`

The ingestor detects sheet types from sheet names and columns, then validates directly from the parsed workbook content in memory.
