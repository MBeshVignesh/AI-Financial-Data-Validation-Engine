# Architecture

```mermaid
flowchart LR
    A["Oracle Smart View Excel Extracts"] --> B["Ingestion Layer<br/>pandas + openpyxl"]
    B --> C["Direct Workbook Parsing<br/>typed in-memory hierarchy frames"]
    C --> D["Validation Agent Workflow"]
    E["Uploaded Target Workbook<br/>curated hierarchy export"] --> D
    F["Mapping Workbook"] --> G["RAG Index<br/>Chroma + local Nomic embeddings"]
    H["Validation Rules Workbook"] --> G
    I["Transformation Notes"] --> G
    J["Prior Exception Log"] --> G
    G --> D
    D --> K["Validation Engine<br/>explicit dataframe checks"]
    K --> L["Exception Report JSON"]
    K --> M["Markdown Report"]
    D --> N["SQLite Run Registry"]
    D --> O["Ollama Reasoner<br/>optional, fallback templating"]
    P["FastAPI"] --> D
    Q["Streamlit Demo"] --> D
```

## Flow

1. Uploaded source and target workbooks are parsed directly into in-memory hierarchy frames, mappings, and rules.
2. Mapping records, rules, transformation notes, and prior exceptions are embedded into a Chroma index using the local `nomic-ai/nomic-embed-text-v1.5` model by default.
3. The agent retrieves relevant context, chooses enabled rules for each dimension, and runs explicit validation checks.
4. Results are saved as JSON and markdown reports, while SQLite stores report metadata for lookup by run ID.

## Design Notes

- The validation layer is rule-based and inspectable so business logic stays transparent.
- RAG augments explanations and rule selection context instead of replacing deterministic checks.
- Ollama is optional at runtime; the app falls back to template-based business summaries when the local model is unavailable.
