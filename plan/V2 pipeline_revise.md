# V2 Pipeline Redesign

## 1. 이 문서의 목적

이 문서는 현재 V2 검색 파이프라인을 어떤 방향으로 수정할지 설명하는 재설계 문서다.

기존 [`V2 pipeline.md`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/plan/V2%20pipeline.md)가
`현재 구현되어 있는 V2의 구조`를 설명하는 문서라면,
이 문서는
`왜 그 구조를 바꿔야 하는지`,
`어떤 구조로 바꿀지`,
`무엇을 새로 만들고 무엇을 그대로 재사용할지`
를 명확히 정리하는 문서다.

이 문서만 읽어도 아래를 이해할 수 있어야 한다.

- 현재 V2의 핵심 문제가 무엇인지
- 새 V2가 어떤 검색 철학을 따르는지
- query parser와 ranking이 어떻게 바뀌는지
- must-have와 soft preference를 어떤 기준으로 나누는지
- 어떤 모듈을 수정하고 어떤 모듈을 새로 만들지
- 구현을 어떤 순서로 진행하면 되는지

## 2. 현재 V2의 문제

현재 V2는 겉으로 보면 태그 기반 검색처럼 보인다.
하지만 실제 ranking의 중심은 여전히 `SigLIP2 text embedding + cosine similarity`이다.

즉 지금 구조는 다음과 같은 문제가 있다.

1. query의 의도를 직접 비교하지 않는다  
사용자가 `black relaxed trousers`를 입력해도,
실제로는 `trousers`, `black`, `relaxed`를 feature별로 강하게 비교하는 것이 아니라
전체 문장을 embedding으로 바꿔 문서 embedding과 cosine similarity를 비교한다.

2. category mismatch를 강하게 제어하지 못한다  
`black`만 맞는 `dress`가
`black trousers`보다 위에 뜰 수 있다.
이건 패션 검색 관점에서 분명한 오답에 가깝다.

3. explanation과 ranking logic가 완전히 연결되어 있지 않다  
현재 explanation은 태그를 비교해서 설명을 만들지만,
실제 순서는 cosine similarity가 정하고 있다.
즉 "왜 이 결과가 위에 왔는지"를 시스템이 일관되게 설명하지 못한다.

4. rule-based query parser의 한계가 있다  
현재 parser는 vocabulary 기반 문자열 탐색에 가깝다.
간단한 query는 처리할 수 있지만,
표현이 조금만 유연해져도 feature 분리가 불안정해질 수 있다.

정리하면,
현재 V2는 `태그 기반처럼 보이는 embedding retrieval`이고,
우리가 만들고 싶은 것은 `태그 기반 symbolic retrieval`이다.

## 3. 새 V2의 핵심 방향

재설계 후 V2는 아래 원칙을 따른다.

1. 검색의 중심은 `사용자가 말한 의도`다  
이미지가 전체적으로 비슷해 보이는가보다
사용자가 요구한 `category`, `color`, `silhouette`, `mood`가 맞는지가 더 중요하다.

2. 1차 retrieval은 `tag match`로 수행한다  
첫 순위 결정은 embedding이 아니라 태그 일치도 점수로 한다.

3. embedding은 보조 역할만 한다  
embedding을 쓰더라도 1차 retrieval 이후의 optional rerank 또는 tie-break 수준에 둔다.

4. parser와 ranker는 같은 feature 구조를 공유한다  
query parser가 뽑은 feature와
archive document가 가진 feature를
같은 schema와 같은 중요도 체계로 비교해야 한다.

5. explanation은 점수 계산과 직접 연결되어야 한다  
결과 설명은 "잘 맞아서 왔다" 수준이 아니라
어떤 속성이 필수로 맞았고,
어떤 속성이 부족했고,
어떤 속성 때문에 감점되었는지를 보여줘야 한다.

한 줄로 요약하면:

`새 V2는 LLM parser가 query를 구조화하고, tag-based ranker가 1차 ranking을 수행하며, explanation이 그 점수 체계를 그대로 반영하는 구조다.`

## 4. 무엇이 바뀌는가

이번 수정의 본질은 아래 네 가지다.

### 4-1. Query parsing

- 기존: rule-based parser
- 변경: LLM parser

기존 parser는
미리 정의된 vocabulary를 기준으로 query 안에서 문자열을 찾는 방식이었다.

새 parser는
자연어 query를 읽고 아래 같은 structured JSON으로 변환한다.

- `canonical_tags`
- `required_features`
- `preferred_features`
- `raw_phrases`
- `confidence`

즉 query를 하나의 문장으로 embedding하는 것이 아니라
검색 가능한 구조화된 feature 세트로 바꾸는 것이 목적이다.

### 4-2. Retrieval core

- 기존: embedding-first
- 변경: tag-match-first

기존에는 query와 document를 embedding space에서 비교한 뒤
cosine similarity로 정렬했다.

새 구조에서는 query와 document의 태그를 직접 비교하여 점수를 만든다.

예를 들면:

- category exact match
- color exact match
- silhouette overlap
- mood mismatch
- required feature missing
- contradiction

이런 단위 점수들의 합으로 최종 순위를 정한다.

### 4-3. Embedding의 역할

- 기존: 주 ranking score
- 변경: optional 보조 score

embedding은 완전히 버리는 것이 아니라,
필요하면 나중에 top 50 또는 top 100 후보에 대해서만 보조 rerank로 사용한다.

하지만 MVP에서는 1차 ranking에서 embedding을 제거한다.

### 4-4. Output explanation

- 기존: 태그 비교 설명이 따로 존재
- 변경: ranking score와 직접 연결된 설명

새 explanation은 아래를 보여줘야 한다.

- 어떤 feature가 필수 매치였는지
- 어떤 feature가 선호 매치였는지
- 어떤 feature가 빠졌는지
- 어떤 feature가 contradiction이라 감점되었는지

즉 explanation은 ranking의 사후 요약이 아니라
ranking의 근거를 그대로 노출하는 역할을 한다.

## 5. 새 V2의 전체 구조

새 구조는 아래 4단계로 이해하면 된다.

### Step 1. LLM Query Parser

모듈 예시:

- `switch_query/v2/llm_parser.py`

입력:

- 자연어 query

출력:

- `canonical_tags`
- `required_features`
- `preferred_features`
- `raw_phrases`
- `confidence`

역할:

- query를 검색 가능한 feature schema로 변환한다
- 어떤 feature가 필수인지, 어떤 feature가 선호인지 구분한다
- 후속 ranker가 직접 사용할 수 있는 정규화된 구조를 만든다

### Step 2. Tag Ranker

모듈 예시:

- `switch_query/v2/tag_ranker.py`

입력:

- parsed query
- archive document list

출력:

- feature match score가 계산된 ranked results

역할:

- query feature와 document feature를 직접 비교한다
- exact / partial / missing / contradiction 판정을 수행한다
- top-k 결과를 점수 기반으로 정렬한다

핵심:

- `document.vector` 없이도 1차 retrieval이 가능해야 한다

### Step 3. Optional Reranker

모듈 예시:

- `switch_query/v2/reranker.py`

역할:

- 1차 retrieval 결과에 대해서만 보조 재정렬을 수행한다
- embedding rerank 또는 LLM rerank를 붙일 수 있다

단,
MVP에서는 사용하지 않는다.

### Step 4. Explanation Builder

기존 `explanation.py`를 재사용하되,
점수 계산 규칙과 직접 연결되게 수정한다.

역할:

- matched features
- preferred matches
- missing required features
- contradictions
- total score breakdown

을 사람이 읽을 수 있는 문장으로 정리한다.

## 6. Parser는 어디까지 할 것인가

이번 재설계에서 parser는 매우 중요하지만,
parser에게 너무 많은 역할을 주면 오히려 시스템이 불안정해진다.

따라서 parser의 역할은 아래로 제한한다.

### parser가 해야 하는 일

- query에서 canonical feature를 추출한다
- raw phrase를 함께 남긴다
- feature를 `required`와 `preferred`로 나눈다
- 전체 해석에 대한 confidence를 준다

### parser가 하지 않아야 하는 일

- 최종 ranking을 직접 판단하지 않는다
- query rewrite 문장을 ranking의 핵심 입력으로 만들지 않는다
- 설명 문장을 생성의 중심에 두지 않는다

즉 parser의 핵심 역할은
`자연어를 structured retrieval schema로 바꾸는 것`
이다.

## 7. must-have와 soft preference를 나누는 기준

이 구분은 새 V2에서 매우 중요하다.

기준은 간단하다.

`이 속성이 틀리면 사용자가 "이건 아예 다른 결과다"라고 느끼는가?`

이 질문에 대한 답으로 나눈다.

### must-have

틀리면 결과 자체가 오답처럼 보이는 속성이다.

예:

- `category`
- query에 명시된 `color`

예를 들어 사용자가 `black trousers`를 찾는데
`black dress`가 나오는 것은
색은 맞아도 category가 틀렸기 때문에 사실상 오답이다.

### soft preference

맞으면 더 좋지만,
안 맞아도 아직 후보로 검토할 수 있는 속성이다.

예:

- `silhouette`
- `mood`
- `material`
- `pattern`
- `texture`
- `era`
- `detail`

예를 들어 `minimal black trousers`에서
`minimal`은 중요하지만
조금 덜 minimal한 trousers가 후보에 남는 것은 가능하다.

### 실무 기준으로 고정할 기본 정책

초기 버전은 아래 정책으로 고정한다.

- `category`: 항상 `required`
- `color`: query에 명시되면 `required`
- `silhouette`, `mood`: 기본 `preferred`
- `material`, `pattern`, `texture`, `era`, `detail`: 있으면 `preferred`, 없으면 무시

예:

- query: `black relaxed trousers with minimal mood`
- required = `category`, `color`
- preferred = `silhouette`, `mood`

이 구조를 넣는 이유는 단순하다.

`black`만 맞는 `dress`를 위로 올리지 않기 위해서다.

## 8. Ranking은 어떻게 바뀌는가

새 V2의 ranking은 cosine similarity 하나가 아니라
feature별 match score의 합으로 결정된다.

### 8-1. feature 중요도

모든 feature가 같은 중요도를 가지면 안 된다.

초기 우선순위는 아래처럼 둔다.

- Tier 1: `category`, `color`
- Tier 2: `silhouette`, `mood`
- Tier 3: `material`, `pattern`, `texture`, `era`
- Tier 4: `detail`

이 우선순위는 parser와 ranker가 공유해야 한다.

### 8-2. 판정 방식

multi-value feature는 문자열 전체 비교가 아니라 set 비교로 처리한다.

각 feature는 아래 네 가지 중 하나로 판정된다.

- `exact`
- `partial`
- `missing`
- `contradiction`

예:

- query `trousers`
- doc `shirt|trousers|shoes` -> `partial`
- doc `dress|heels` -> `contradiction`
- doc value 없음 -> `missing`

### 8-3. 초기 점수 규칙

첫 버전은 learned ranking이 아니라
설명 가능한 hand-tuned scoring으로 시작한다.

초기 추천 점수:

- `category exact`: `+8`
- `category partial overlap`: `+4`
- `color exact`: `+6`
- `silhouette exact`: `+4`
- `mood exact`: `+3`
- `material/pattern/texture/era exact`: `+2`
- `detail overlap`: `+1`
- `required missing`: `-10`
- `category contradiction`: `-12`
- `color contradiction`: `-8`
- `preferred missing`: `0`

핵심 원칙:

- category는 가장 강한 축이다
- color는 중요하지만 category보다 아래다
- preferred missing은 과도하게 벌점 주지 않는다
- contradiction은 missing보다 더 강하게 벌점 준다

## 9. Hard filter는 어디에 둘 것인가

모든 feature를 hard filter로 걸면 recall이 지나치게 줄어든다.

따라서 초기 정책은 아래처럼 단순하게 가져간다.

### hard filter를 두는 경우

- query에 `category`가 있을 때
- candidate document와 category overlap이 `0`이면 제외

### hard filter를 두지 않는 경우

- `color`
- `silhouette`
- `mood`
- 그 외 모든 feature

이 경우는 제외하지 않고 penalty만 준다.

이유:

- category는 retrieval intent의 중심축이다
- color는 tagging noise와 누락 가능성이 있어 penalty가 더 안전하다
- silhouette, mood는 해석이 흔들릴 수 있어 hard filter로 쓰기 위험하다

즉 정책은 이렇다.

`hard filter는 category에만 두고, 나머지는 score로 제어한다.`

## 10. 무엇을 재사용하고 무엇을 바꾸는가

이번 재설계는 전처리 자산을 버리는 작업이 아니다.
오히려 preprocessing에서 만든 결과를 더 잘 활용하기 위한 구조 변경이다.

### 그대로 재사용하는 것

- `normalized_tags.csv`
- `canonical_mapping_final.csv`
- `v2_documents.json`
- `V2ArchiveDocument`
- preprocessing pipeline
- 기존 explanation의 기본 뼈대

즉
이미지 태깅과 canonical mapping으로 만들어 둔 자산은 그대로 살린다.

### 약화하거나 제거하는 것

- rule-based query parsing 중심 구조
- `query_document -> text encoder -> cosine only` 중심 ranking
- parser가 feature vocabulary를 단순 문자열 탐색하는 방식

이 부분들은 완전히 삭제해도 되지만,
초기에는 fallback 또는 디버깅용으로 약하게 남겨 둘 수 있다.

## 11. 구현 후 시스템 흐름

재설계 후 실행 흐름은 아래와 같다.

1. 사용자가 query를 입력한다
2. LLM parser가 query를 structured schema로 변환한다
3. tag ranker가 archive documents와 feature별 비교를 수행한다
4. category hard filter와 feature weights로 top-k를 정한다
5. explanation builder가 점수 근거를 사람이 읽을 수 있는 형태로 만든다
6. 필요하면 이후에만 optional rerank를 붙인다

즉 런타임 구조는 아래처럼 요약된다.

`natural language query -> structured query -> tag scoring -> top-k -> explanation`

## 12. 구현 전에 최종 확정할 사항

아래 항목은 구현 전에 문서 기준으로 확정한다.

### 12-1. LLM provider

최초 구현은 `Saltlux Luxia chat API`를 사용한다.

이유:

- 현재 목표는 자유 생성이 아니라 query를 structured feature schema로 파싱하는 것이다
- 한국어 query 해석이 중요한 환경에서 Luxia를 우선 실험하는 것이 자연스럽다
- parser는 reasoning보다 instruction following과 intent comprehension이 더 중요하므로 Luxia의 `llm` 계열 모델이 잘 맞는다

단,
Luxia API가 OpenAI Structured Outputs처럼 schema를 강제 보장하는 것으로 가정하지는 않는다.

따라서 parser 운영 원칙은 아래처럼 둔다.

- Luxia는 JSON 형식으로 응답하도록 강하게 프롬프트한다
- 응답은 로컬 schema validation으로 반드시 검증한다
- validation 실패 시 retry를 수행한다
- 반복 실패 시 rule-based fallback 또는 parser failure로 처리한다

권장 모델:

- 운영 기본 모델: `luxia3-llm-32b-0731`
- 빠른 실험용 모델: `luxia3-llm-8b-0731`

정리:

- provider는 인터페이스로 분리한다
- 최초 구현은 Luxia를 사용한다
- schema 안정성은 API에 전적으로 기대하지 않고 애플리케이션 레이어에서 보강한다

### 12-2. Parser output schema

첫 버전 parser는 아래 필드만 반환한다.

- `canonical_tags`
- `required_features`
- `preferred_features`
- `raw_phrases`
- `confidence`

이 이상을 초기에 과도하게 넣지 않는다.

### 12-3. Rerank policy

MVP에서는 rerank를 사용하지 않는다.

이유:

- 1차 retrieval 품질 문제를 먼저 명확히 해결해야 한다
- rerank를 미리 넣으면 문제 원인 분리가 어려워진다

이후 필요하면
`top 50` 또는 `top 100`에 대해서만
embedding rerank를 optional하게 추가한다.

## 13. 구현 순서

가장 안전한 구현 순서는 아래다.

1. `V2ParsedQuery` schema 확장
2. `LLMQueryParser` 인터페이스 추가
3. parser JSON schema 고정
4. `TagRanker` 구현
5. `V2Pipeline`을 embedding-first에서 tag-rank-first로 변경
6. explanation을 새 점수 체계와 연결
7. CLI에서 parser mode 선택 가능하게 추가
8. subset 데이터로 query evaluation 수행

중요한 점:

parser만 먼저 바꾸고 ranking을 그대로 두면 안 된다.
이번 수정의 핵심은
`parser schema + ranker scoring schema`를 함께 고정하는 것이다.

## 14. MVP 목표

첫 번째 목표는 아주 명확하다.

`black relaxed trousers with minimal mood`를 입력했을 때,
적어도 `black dress`보다 `black relaxed trousers`가 위에 와야 한다.

이 목표가 의미하는 바는 아래와 같다.

- parser가 category, color, silhouette, mood를 정확히 분리해야 한다
- ranker가 category mismatch를 강하게 제어해야 한다
- explanation이 실제 점수와 일치해야 한다

즉 MVP는 "완벽한 패션 이해"가 아니라
`명백한 검색 오류를 막는 구조`를 먼저 만드는 것이다.

## 15. 최종 요약

이번 V2 재설계는 단순한 모델 교체가 아니다.
검색 철학 자체를 바꾸는 작업이다.

기존 V2가
`embedding similarity가 중심이고 태그는 설명용으로 덧붙는 구조`
였다면,
새 V2는
`태그 기반 의도 검색이 중심이고 embedding은 보조로 물러나는 구조`
다.

최종 확정안은 아래 한 문장으로 요약된다.

`OpenAI structured parser가 query를 required/preferred feature로 분해하고, category 중심 tag ranker가 1차 retrieval을 수행하며, explanation이 그 점수 근거를 그대로 보여주고, rerank는 MVP에서 사용하지 않는다.`
