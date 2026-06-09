# JustPutBasket

패션 아카이브 이미지 검색을 실험하기 위한 Python 프로토타입입니다. 사용자가 자연어로 원하는 옷차림이나 무드를 입력하면, 이미지별 태그와 아이템 정보를 바탕으로 관련성이 높은 아카이브 이미지를 찾아 순위를 매깁니다.

## What It Does

- 이미지 태그 CSV를 전처리해 검색용 문서로 변환합니다.
- V1, V2, V3 단계의 검색 파이프라인을 포함합니다.
- V3는 쿼리를 아이템 단위로 해석하고, 심볼릭 매칭과 임베딩 후보 검색을 함께 사용합니다.
- 검색 결과를 CSV 또는 HTML 리포트로 저장할 수 있습니다.
- OpenAI/Luxia 기반 태깅, 로컬 VLM 태깅, Fashionpedia 변환용 보조 도구가 포함되어 있습니다.

## Project Structure

```text
switch_query/
  v1/             초기 검색 파이프라인
  v2/             태그 랭킹과 문서 기반 검색 파이프라인
  v3/             아이템 인식 검색 파이프라인
  tagging/        이미지 태깅 및 태그 정규화 도구
  image_module/   이미지 처리/태깅 실험 모듈
tests/            파이프라인과 CLI 테스트
```

## Setup

Python 3.11 이상이 필요합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

선택 기능을 사용할 때는 필요한 extra를 함께 설치합니다.

```bash
pip install -e ".[baseline]"
pip install -e ".[local_vlm]"
pip install -e ".[v1]"
```

## Basic V3 Workflow

1. 정규화된 태그 CSV에서 V3 입력 JSONL을 만듭니다.

```bash
python -m switch_query.v3.preprocessing_cli build-inputs \
  --normalized-tags data/normalized_tags.csv \
  --mode full
```

2. 아이템 추출 결과와 입력을 병합해 아카이브 문서를 만듭니다.

```bash
python -m switch_query.v3.preprocessing_cli merge-documents \
  --inputs data/v3/item_inputs_full.jsonl \
  --outputs data/v3/item_outputs_full.jsonl \
  --output data/v3/archive_documents.jsonl
```

3. 검색 인덱스를 생성합니다.

```bash
python -m switch_query.v3.cli build-index \
  --documents data/v3/archive_documents.jsonl \
  --output data/v3/archive_index.json
```

4. 자연어 쿼리로 검색합니다.

```bash
python -m switch_query.v3.cli run-query \
  --index data/v3/archive_index.json \
  --query "black leather jacket with slim denim" \
  --html-output tmp/results.html
```

## Tests

```bash
python -m pytest
```

## Notes

실험용 데이터, 생성 산출물, 임시 파일은 `data/`와 `tmp/` 아래에 두는 것을 권장합니다. API 기반 태깅이나 추출 기능은 별도의 API 키와 모델 접근 권한이 필요할 수 있습니다.
