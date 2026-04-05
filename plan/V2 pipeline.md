# V2 Pipeline

## 한 줄 요약

V2는 `이미지가 얼마나 비슷한가`보다 `사용자가 무엇을 원한다고 말했는가`를 더 중요하게 보는 검색 파이프라인이다. 현재는 이미 이 파이프라인 내용을 기반으로 구현이 되어 있으며 여기서의 개선 사항이 V2 pipeline_revise.md에 기록한다. 내가 너한테 명령하는 수정사항들은 이 문서 말고 이제 V2 pipeline_revise.md를 살펴봐라.

쉽게 말하면:

- V1은 `비슷하게 생긴 룩`을 잘 찾는 쪽
- V2는 `검은색, tailored, minimal 같은 의도`를 잘 찾는 쪽

이 문서는 지금 우리가 실제로 구현하고 있는 V2 흐름을 쉬운 말로 정리한 문서다.

## 왜 V2가 필요한가

사용자가 아래처럼 검색한다고 가정하자.

`black tailored coat with minimal but sharp mood`

이때 정말 중요한 것은:

- 검은색인가
- tailored한가
- minimal but sharp한 분위기인가

이지, 단순히 사진이 비슷해 보이느냐만은 아니다.

그래서 V2는 이미지를 먼저 `태그`로 바꾸고, 그 태그를 이용해 검색한다.

## V2의 큰 흐름

V2는 아래 순서로 동작한다.

1. 아카이브 이미지 목록을 만든다.
2. 이미지들을 AI로 태깅한다.
3. 태그를 정리해서 같은 뜻끼리 묶는다.
4. 정리된 태그로 검색용 문서를 만든다.
5. 사용자의 검색어도 같은 규칙으로 해석한다.
6. 둘을 비교해서 top 20 결과를 돌려준다.
7. 왜 이 결과가 나왔는지 explanation을 함께 보여준다.

## 핵심 아이디어

V2는 아래 원칙을 따른다.

1. 검색의 중심은 `텍스트 의도`다.
2. 이미지 태깅은 미리 offline으로 해 둔다.
3. 검색할 때는 빠르게 실행돼야 한다.
4. 태그는 `raw 표현`과 `canonical 표현`을 둘 다 보존한다.
5. explanation이 가능해야 한다.

## 자주 나오는 용어

### inventory

아카이브 안에 어떤 이미지가 있는지 정리한 목록이다.

예:

- image_id
- file_path
- brand
- year
- season_group

### raw tag

AI가 이미지를 보고 처음 붙인 태그다.

예:

- `jet black`
- `sharp tailoring`
- `minimal but sharp`

아직 표현이 제각각일 수 있다.

### canonical tag

검색이 잘 되도록 raw tag를 정리한 값이다.

예:

- `jet black` -> `black`
- `sharp tailoring` -> `tailored`
- `minimal but sharp` -> `minimal|sharp`

### draft

초안이라는 뜻이다.

`canonical_mapping_draft.csv`는 시스템이 자동으로 만든 첫 번째 매핑 초안이다.
사람이 이 파일을 보고 고친 뒤 `canonical_mapping_final.csv`로 확정한다.

## 현재 고정 schema

V2는 아래 속성을 중심으로 태깅하고 검색한다.

- `category`
- `silhouette`
- `color`
- `material`
- `pattern`
- `texture`
- `mood`
- `season`
- `era`
- `detail`

## 실제 운영 방식

현재 우리가 정한 운영 방식은 아래와 같다.

### 1. sample-first는 일반 API로 빠르게 확인

처음부터 1만 장 넘는 전체 이미지를 다 태깅하지 않는다.

먼저 브랜드별 샘플 이미지를 뽑아서:

1. 태깅이 잘 되는지
2. 어떤 표현들이 나오는지
3. canonical mapping을 어떻게 잡아야 하는지

를 본다.

이 단계는 디버깅이 중요하므로 `OpenAI sync API`로 돌린다.

현재 기본 모델:

- `gpt-4.1-mini`

### 2. full archive는 Batch API로 처리

전체 이미지는 수가 많기 때문에, 한 장씩 일반 API로 돌리면 느리고 비쌀 수 있다.

그래서 full archive는 `OpenAI Batch API`로 제출하고, 나중에 결과를 한 번에 수집한다.

이 방식의 장점:

- 비용 절감
- 대량 처리에 적합
- sample 단계와 full 단계의 목적을 분리할 수 있음

## 현재 전처리 산출물

전처리 결과는 아래 폴더에 저장된다.

`tmp/v2_preprocessing/<dataset_slug>/`

예를 들어 현재 데이터셋이라면:

`tmp/v2_preprocessing/data__2026__spring-ready-to-wear/`

이 안에 아래 파일들이 생긴다.

- `inventory.csv`
- `sample_manifest.csv`
- `raw_tags_sample.csv`
- `frequency_sample.csv`
- `canonical_mapping_draft.csv`
- `raw_tags_full.csv`
- `frequency_full.csv`
- `canonical_mapping_final.csv`
- `normalized_tags.csv`
- `batch_input_full.jsonl`
- `batch_job_full.json`
- `batch_output_full.jsonl`
- `batch_errors_full.jsonl`

## 단계별 설명

### 1. Inventory 만들기

목적:

- 아카이브 안에 있는 collection 이미지 목록을 만든다.

출력:

- `inventory.csv`

이 파일은 이후 모든 단계의 출발점이다.

### 2. Sample-first 태깅

목적:

- 적은 수의 샘플 이미지로 태깅 결과를 먼저 확인한다.

출력:

- `sample_manifest.csv`
- `raw_tags_sample.csv`
- `frequency_sample.csv`
- `canonical_mapping_draft.csv`

여기서 중요한 것은 `canonical_mapping_draft.csv`다.
이 파일을 보면 어떤 raw 표현이 많이 나오는지 알 수 있고, 어떤 것들을 묶어야 하는지 판단할 수 있다.

### 3. Canonical mapping 확정

목적:

- 검색용으로 쓸 표준 표현을 정한다.

예:

- `jet black` -> `black`
- `scarlet` -> `red`
- `gown` -> `dress`

이 단계는 반자동이다.

즉:

1. 시스템이 초안을 만든다.
2. 사람이 검토한다.
3. 최종본을 `canonical_mapping_final.csv`로 저장한다.

### 4. Full archive batch tagging

목적:

- 아카이브 전체 이미지를 태깅한다.

방식:

1. batch input 파일 생성
2. OpenAI Batch API에 제출
3. 완료 후 결과 수집
4. `raw_tags_full.csv`와 `frequency_full.csv` 생성

이 단계는 오래 걸릴 수 있으므로 `submit`과 `collect`를 분리해 두었다.

### 5. Normalize

목적:

- raw tag와 canonical tag를 함께 가진 최종 테이블을 만든다.

출력:

- `normalized_tags.csv`

이 파일이 매우 중요하다.
이제부터 V2 retrieval은 이 파일을 기반으로 움직인다.

## 왜 raw와 canonical을 둘 다 저장하나

둘 다 필요하기 때문이다.

- `raw_*`는 사람이 이해하기 쉽고, explanation에도 유용하다.
- `canonical_*`는 검색할 때 표현 차이 때문에 놓치지 않게 해준다.

예를 들어:

- raw: `minimal but sharp`
- canonical: `minimal|sharp`

이렇게 하면 검색은 안정적으로 하고, 설명은 자연스럽게 할 수 있다.

## 이후 Retrieval 단계

전처리가 끝나면 그다음은 검색 단계다.

흐름은 아래와 같다.

1. `normalized_tags.csv`를 읽는다.
2. 각 이미지에 대해 검색용 document를 만든다.
3. document embedding을 만든다.
4. 사용자의 query를 canonical tag로 해석한다.
5. query도 embedding한다.
6. 가장 비슷한 top 20 이미지를 찾는다.
7. explanation을 만든다.

## Explanation은 어떻게 만들까

V2는 단순히 점수만 주지 않고, 왜 추천했는지도 보여주려고 한다.

기본 구조:

- `matched_attributes`
- `mismatched_attributes`
- `missing_attributes`

예:

- matched: `color=black`, `silhouette=tailored`
- mismatched: `material expected wool but candidate is unclear`
- missing: `era`

이렇게 하면 결과를 사람이 검토하기 쉬워진다.

## user uploaded image는 어떻게 쓰나

지금 단계에서는 입력 필드로는 남겨두되, 검색 점수의 중심으로 쓰지 않는다.

이유:

- V2가 다시 V1처럼 `visual similarity` 중심으로 기울지 않게 하기 위해서다.
- V2는 끝까지 `intent-first retrieval`이어야 한다.

즉 현재 기준으로는:

- `query_text`가 핵심
- `user_uploaded_image`는 나중에 약한 힌트로 붙일 수 있음

## 현재 구현 기준 개발 순서

지금 시점에서 우리가 실제로 밟는 순서는 아래와 같다.

### Phase 1. 전처리 준비

할 일:

1. dataset 경로 확인
2. inventory 생성
3. sample manifest 생성

### Phase 2. sample-first 확인

할 일:

1. sync API로 sample 태깅
2. raw tag 확인
3. frequency 확인
4. draft mapping 확인

### Phase 3. canonical mapping 확정

할 일:

1. draft를 복사해 final 파일 생성
2. synonym과 parent_map 정리
3. 너무 과하게 합치지 않도록 검토

### Phase 4. full archive batch 실행

할 일:

1. batch submit
2. 완료 후 batch collect
3. raw_tags_full.csv 생성
4. frequency_full.csv 생성

### Phase 5. normalize

할 일:

1. canonical_mapping_final.csv 적용
2. normalized_tags.csv 생성

### Phase 6. retrieval

할 일:

1. normalized tags를 document로 변환
2. index 생성
3. query parser 실행
4. top 20 retrieval
5. explanation 생성

## 학부생 버전으로 이해하면

이 프로젝트를 아주 단순하게 보면 아래와 같다.

1. 이미지를 AI에게 설명하게 만든다.
2. 비슷한 뜻의 표현을 사람이 정리한다.
3. 정리된 설명을 검색용 데이터로 저장한다.
4. 사용자의 문장을 같은 규칙으로 해석한다.
5. 둘을 비교해서 가장 잘 맞는 이미지를 찾는다.

즉 V2는 결국:

`이미지 -> 태그 -> 정리 -> 검색`

의 흐름이다.

## 지금 기준으로 가장 중요한 파일

- [`switch_query/v2/preprocessing.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/preprocessing.py)
- [`switch_query/v2/preprocessing_cli.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/v2/preprocessing_cli.py)
- [`switch_query/tagging/openai_vlm_tagger.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/tagging/openai_vlm_tagger.py)
- [`switch_query/tagging/openai_batch_tagger.py`](/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/switch_query/tagging/openai_batch_tagger.py)

## 마지막 요약

현재 V2의 실제 전략은 아래 한 문장으로 정리할 수 있다.

`샘플 이미지는 OpenAI sync API로 빠르게 확인하고, 전체 아카이브는 OpenAI Batch API로 태깅한 뒤, 사람이 canonical mapping을 확정해서 normalized_tags.csv를 만들고, 그 파일로 intent-first retrieval을 수행한다.`
