# QuerySwitch Max

QuerySwitch Max is the current product direction for this repository. The target
workflow is image-first design exploration, not recommendation-first search.

```text
Original Query -> Reference Feed -> Design Baskets -> Basket Graph -> Pruning -> Generation Run
```

The repository still contains earlier V1, V2, and V3 search-pipeline
experiments. Those modules are useful as retrieval and tagging foundations, but
they should be treated as supporting work for QuerySwitch Max rather than the
main product story.

## Product Direction

QuerySwitch Max is built around Reference Baskets:

1. A user enters one Original Query.
2. The system prepares a Reference Feed of visual candidates.
3. The user collects Reference Images into Design Baskets.
4. Each Design Basket represents a user-authored visual direction.
5. A Basket Graph summarizes observed visual attributes and inferred design
   intent.
6. The user prunes Graph Nodes by keeping or removing them.
7. A Generation Run uses the kept graph snapshot to generate basket-specific
   images.

## Current Repository Structure

```text
switch_query/
  queryswitch_max/   QuerySwitch Max package location
  image_module/      Image processing, archive enrichment, and retrieval modules
  tagging/           Image tagging and tag normalization tools
  v1/                Early search pipeline
  v2/                Tag-ranking and document-based search pipeline
  v3/                Item-aware search pipeline

tests/               Pipeline, CLI, tagging, and compatibility tests
```

## Module Roles

### QuerySwitch Max

`switch_query/queryswitch_max/` is reserved for the QuerySwitch Max application
layer: Reference Feed, Design Baskets, Basket Graphs, Pruning, Generation Runs,
and the API/UI contracts that support that workflow.

### Image Module

`switch_query/image_module/` supports the Reference Feed and archive-based image
retrieval path. It includes image/archive models, enrichment, retrieval helpers,
and pipeline logic that can feed QuerySwitch Max.

### Legacy Search Pipelines

`switch_query/v1/`, `switch_query/v2/`, and `switch_query/v3/` are earlier
retrieval experiments. They are not the current user-facing product direction.
Use them as implementation references or migration sources when they help the
QuerySwitch Max workflow.

## Local Setup

Python 3.11 or newer is required.

```bash
uv sync
```

If you are not using `uv`, install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras are available for provider-specific or local model workflows:

```bash
pip install -e ".[baseline]"
pip install -e ".[local_vlm]"
pip install -e ".[v1]"
```

## Running Tests

Run the full test suite:

```bash
uv run pytest
```

Or run a specific pipeline area:

```bash
uv run pytest tests/test_v3_*.py
uv run pytest tests/test_image_module*.py
```

## Image Module Example

The image module can still be exercised directly while QuerySwitch Max is being
connected around it:

```bash
uv run python -m switch_query.image_module.cli_pipeline \
  --query-text "bohemian summer white cotton dress mood board" \
  --stage mood_board \
  --parser-backend llm_with_fallback \
  --final-top-k 10 \
  --vlm-reranker-top-n 10 \
  --output-format html \
  --html-output-path data/cache/queryswitch_report.html
```

## Terminology

Use QuerySwitch Max product terms when describing new behavior:

- Original Query
- Reference Feed
- Reference Image
- Design Basket
- Basket Graph
- Graph Node
- Pruning
- Generation Run

Avoid presenting the project as only a V1/V2/V3 recommendation or ranking
pipeline. Those pieces are supporting modules for the Reference Basket product
direction.
