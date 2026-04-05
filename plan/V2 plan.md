# V2 Plan

## 1. 문서 목적

이 문서는 [`V2 pipeline_revise.md`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/plan/V2%20pipeline_revise.md)에 정리된 재설계 방향을
실제 코드 수정 작업으로 번역한 구현 계획서다.

이 문서의 목적은 단순히 방향을 설명하는 것이 아니라,
구현자가 바로 작업에 들어갈 수 있도록 아래를 결정 완료 상태로 정리하는 것이다.

- 무엇을 수정해야 하는가
- 왜 그 부분을 수정해야 하는가
- 어떤 방식으로 수정할 것인가
- 어떤 순서로 구현할 것인가
- 어떤 테스트로 완료를 판단할 것인가

즉 이 문서는 `구조 설명 문서`가 아니라 `실행 계획 문서`다.

## 2. 목표와 성공 기준

이번 작업의 목표는 현재 `embedding-first V2`를 `tag-match-first V2`로 전환하는 것이다.

기존 V2는 겉으로는 태그 검색처럼 보이지만,
실제 ranking은 `SigLIP2 text embedding + cosine similarity`에 크게 의존하고 있다.
이번 수정에서는 그 중심을 바꿔서,
query와 문서를 `feature 단위`로 직접 비교하는 symbolic retrieval 구조를 만든다.

### 성공 기준

이번 재설계가 완료되었다고 판단하는 기준은 아래와 같다.

1. query parser가 자연어 query를 structured feature schema로 변환할 수 있어야 한다
2. 1차 retrieval은 cosine similarity가 아니라 tag score로 수행되어야 한다
3. category mismatch가 ranking에서 강하게 제어되어야 한다
4. explanation이 ranking logic와 직접 연결되어야 한다
5. CLI와 테스트가 모두 새 구조를 반영해야 한다

### MVP 기준

MVP 성공 기준은 아래 한 문장으로 요약된다.

`black relaxed trousers with minimal mood` 같은 query에서 `black dress`보다 `black trousers` 계열 결과가 위에 와야 한다.

## 3. 기본 전제

이번 계획은 아래 전제를 따른다.

- 새 문서는 `plan/V2 plan.md`로 만든다
- 기존 [`plan/V2 pipeline.md`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/plan/V2%20pipeline.md)는 현재 구현 구조 설명 문서로 유지한다
- 기존 [`plan/V2 pipeline_revise.md`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/plan/V2%20pipeline_revise.md)는 방향 문서로 유지한다
- `V2 plan.md`는 실제 구현 실행 계획 문서 역할을 한다

## 4. 현재 코드 기준 수정 대상

아래 파일들은 이번 재설계에서 직접적인 수정 대상이 된다.

### 4-1. Query / Result Schema

- [`switch_query/v2/models.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/models.py)

역할:

- `V2ParsedQuery` schema 확장
- `V2RankedResult` schema 확장
- 필요한 metadata 필드 추가

### 4-2. Parser

- [`switch_query/v2/parser.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/parser.py)
- 새 `switch_query/v2/llm_parser.py`

역할:

- 기존 rule-based parser는 fallback 또는 보조 parser로 축소
- Luxia 기반 structured parser 추가

### 4-3. Ranking

- 새 `switch_query/v2/tag_ranker.py`
- [`switch_query/v2/pipeline.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/pipeline.py)

역할:

- exact / partial / missing / contradiction 기반 tag ranker 구현
- pipeline 흐름을 `parser -> tag rank -> explanation` 구조로 변경

### 4-4. Explanation

- [`switch_query/v2/explanation.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/explanation.py)

역할:

- 기존 matched / mismatched / missing 설명을
  score-linked explanation으로 개선

### 4-5. Index / Storage / CLI

- [`switch_query/v2/index.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/index.py)
- [`switch_query/v2/cli.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/cli.py)

역할:

- MVP에서 vector 의존을 제거하거나 optional로 격하
- parser mode / provider 설정 반영
- 새 결과 포맷 반영

### 4-6. Tests

- [`tests/test_v2_pipeline.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/tests/test_v2_pipeline.py)
- [`tests/test_v2_cli.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/tests/test_v2_cli.py)

역할:

- 기존 cosine 기반 기대값을 제거
- 새 tag-first 정책 기준으로 테스트 재작성

## 5. 구현 작업 스트림

이번 작업은 파일 단위보다 작업 스트림 단위로 진행하는 것이 안전하다.

### Workstream A. Query / Result Schema 확장

목표:

- parser와 ranker가 같은 schema를 공유하도록 기반 타입을 먼저 정리한다

수정 내용:

- `V2ParsedQuery`에 아래 필드를 추가한다
  - `required_features`
  - `preferred_features`
  - `confidence`
- `canonical_tags`와 `raw_phrases`는 유지한다
- `query_document`는 ranking 핵심 입력이 아니라 optional / fallback 용도로만 남긴다
- `V2RankedResult`에는 아래를 추가한다
 
  - `match_reasons`

의도:

- parser 결과가 ranker로 직접 전달될 수 있어야 한다
- explanation이 단순 문자열이 아니라 구조화된 근거를 가질 수 있어야 한다

### Workstream B. Luxia Parser 도입

목표:

- 자연어 query를 stable한 JSON schema로 변환하는 LLM parser를 도입한다

수정 내용:

- Luxia chat API를 사용하는 parser 모듈을 새로 만든다
- 기본 모델은 `luxia3-llm-32b-0731`로 한다
- 빠른 실험용 옵션은 `luxia3-llm-8b-0731`로 둔다
- 출력 형식은 JSON 하나로 고정한다
- parser는 아래만 반환한다
  - `canonical_tags`
  - `required_features`
  - `preferred_features`
  - `raw_phrases`
  - `confidence`

운영 원칙:

- 응답은 로컬 validation을 통과해야만 사용한다
- validation 실패 시 retry를 수행한다
- 반복 실패 시 rule-based fallback 또는 parser failure 처리한다
- parser는 ranking 결정을 하지 않고 feature extraction만 수행한다

의도:

- parser를 자연어 해석기로 쓰되, ranking logic이 parser 내부로 새지 않게 막는다

### Workstream C. Tag Ranker 구현

목표:

- cosine similarity를 대체할 1차 retrieval engine을 만든다

핵심 원칙:

- multi-value feature는 문자열 전체 비교가 아니라 set 비교로 처리한다
- 판정 타입은 `exact`, `partial`, `missing`, `contradiction` 네 가지로 고정한다
- `partial`은 exact의 절반 점수로 고정한다

초기 점수 규칙:

- `category exact`: `+8`
- `category partial`: `+4`
- `color exact`: `+6`
- `silhouette exact`: `+4`
- `silhouette partial`: `+2`
- `mood exact`: `+3`
- `mood partial`: `+1`
- `material/pattern/texture/era exact`: `+2`
- `detail overlap`: `+1`
- `required missing`: `-10`
- `category contradiction`: `-12`
- `color contradiction`: `-8`
- `preferred missing`: `0`

required / preferred 기본 정책:

- `category`는 항상 required
- `color`는 query에 있으면 required
- `silhouette`, `mood`는 preferred
- `material`, `pattern`, `texture`, `era`, `detail`은 있으면 preferred

hard filter 정책:

- category만 hard filter 대상이다
- category overlap이 `0`이면 제외한다
- color 이하 feature는 점수로만 제어한다

의도:

- 관련 없는 category가 color match만으로 상위에 오르는 문제를 막는다
- exact / partial / contradiction의 의미를 코드 수준에서 명확히 만든다

### Workstream D. Pipeline 재배치

목표:

- 기존 cosine-first pipeline을 tag-first pipeline으로 전환한다

수정 내용:

- 현재 `parser -> query embedding -> cosine ranking` 흐름을 제거한다
- 새 흐름은 `parser -> tag ranker -> explanation`으로 고정한다
- `TextEncoder`는 MVP query path에서 필수가 아니게 만든다
- archive index는 feature vocabulary와 canonical/raw tags만으로 검색 가능해야 한다
- rerank는 구조상 붙일 수 있게 열어 두되 MVP에서는 비활성화한다

의도:

- 새 구조의 중심이 embedding이 아니라 tag score라는 점을 코드 구조에서 명확히 만든다

### Workstream E. Explanation 재작성

목표:

- explanation이 실제 ranking 근거를 그대로 설명하게 만든다

수정 내용:

- 현재 `matched / mismatched / missing` 구조는 유지하되 의미를 더 명확히 정리한다
- explanation에는 최소한 아래가 포함되어야 한다
  - matched required features
  - matched preferred features
  - missing required features
  - contradictions
  - 최종 점수 핵심 근거

설명 원칙:

- explanation 문구는 “왜 이 결과가 위에 왔는가”를 ranking과 동일한 언어로 설명해야 한다
- 현재처럼 cosine score와 분리된 별도 설명이 되면 안 된다

### Workstream F. CLI와 출력 포맷 정리

목표:

- 새 검색 구조를 CLI와 결과 파일에서도 그대로 노출한다

수정 내용:

- `run_query`에 parser provider 또는 parser mode 옵션을 추가한다
- 결과 CSV에는 필요 시 아래를 추가한다
  - score breakdown
  - decision trace
- HTML preview는 새 explanation을 그대로 출력한다
- 기존 encoder / model_name 인자는 MVP에서는 optional 또는 deprecated로 문서화한다
- build index 단계는 vector 없이도 동작하도록 계획한다

의도:

- 새 구조가 내부 코드에만 반영되는 것이 아니라 실제 사용 인터페이스에도 드러나게 만든다

## 6. 테스트 계획

이번 재설계는 ranking 철학이 바뀌는 작업이므로,
테스트도 새 구조 기준으로 다시 정의해야 한다.

### 6-1. Parser 테스트

검증할 것:

- Luxia 응답이 유효 JSON이면 required / preferred가 정확히 채워진다
- validation 실패 응답은 retry 또는 fallback 경로로 들어간다
- parser가 feature extraction만 수행하고 ranking 판단은 하지 않는다

### 6-2. Ranker 테스트

검증할 것:

- exact > partial > missing > contradiction 순서가 보장된다
- `trousers` query에서 `shirt|trousers|shoes`는 partial로 판정된다
- `trousers` query에서 `dress|heels`는 contradiction으로 판정된다
- category hard filter가 category mismatch 문서를 제외한다

### 6-3. Pipeline 테스트

검증할 것:

- cosine similarity 없이도 top-k가 계산된다
- `black tailored coat` query에서 coat가 dress보다 앞선다
- uploaded image는 metadata에만 남고 scoring에는 쓰이지 않는다
- parser 결과가 ranker로 그대로 전달된다

### 6-4. CLI 테스트

검증할 것:

- 결과 CSV가 새 explanation 또는 score breakdown 필드를 반영한다
- HTML preview가 새 explanation을 반영한다
- parser/provider 옵션이 실제 run_query에 반영된다

### 6-5. 회귀 테스트

검증할 것:

- 기존 preprocessing 결과물은 그대로 재사용 가능하다
- `normalized_tags.csv` 기반 흐름이 깨지지 않는다
- index load/save가 새 schema와 호환된다

## 7. 구현 순서

이번 작업은 아래 순서로 진행하는 것이 가장 안전하다.

1. query / result schema 확장
2. Luxia parser 인터페이스와 validation layer 추가
3. tag ranker 구현
4. pipeline을 tag-first 구조로 전환
5. explanation을 새 점수 체계와 연결
6. CLI와 출력 포맷 정리
7. 테스트 교체 및 회귀 검증
8. subset 평가로 relevance 확인

중요한 점:

parser만 먼저 바꾸고 ranking을 그대로 두면 안 된다.
이번 수정의 핵심은
`parser schema + ranker scoring schema`를 함께 고정하는 것이다.

## 8. 구현 기본값

이번 계획은 아래 기본값을 전제로 한다.

- 새 파일 이름은 `plan/V2 plan.md`로 고정한다
- 기존 `plan/V2 pipeline.md`는 덮어쓰지 않는다
- parser provider 기본값은 Luxia다
- MVP에는 rerank를 넣지 않는다
- partial score는 exact의 절반으로 고정한다
- category만 hard filter 대상으로 둔다
- 구현 계획 문서는 한국어로 작성하되, 코드 심볼과 파일 경로는 원문 그대로 적는다

## 9. 최종 요약

이번 `V2 plan.md`의 역할은 단순하다.

[`V2 pipeline_revise.md`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/plan/V2%20pipeline_revise.md)가
`어떤 방향으로 바꿀 것인가`를 설명하는 문서라면,
이 문서는
`그 방향을 실제 코드 수정 작업으로 어떻게 옮길 것인가`
를 설명하는 문서다.

최종적으로 구현자는 이 문서를 기준으로 아래를 수행하게 된다.

- parser를 Luxia 기반 structured parser로 바꾸고
- cosine-first ranking을 tag-first ranking으로 바꾸고
- explanation을 점수와 연결하고
- CLI와 테스트를 새 구조에 맞게 정리한다

즉 이 문서는 이번 V2 재설계의 `실행 체크리스트이자 구현 가이드`다.
