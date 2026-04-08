
1. vintage 같은 mood가 들어간 쿼리에 대해서 제대로 추천될 수 있게 수정
2. multi item에 대해서도 쿼리가 제대로 들어갈 수 있게 수정
3. color에 대한 쿼리가 제대로 들어가야함


## Current Status

### Done
- [x] `style_concepts`를 query/item 레벨에 추가했고, document-level concept support는 item aggregation/inference로 처리한다.
- [x] `vintage`를 다른 style concept와 동일하게 처리하도록 바꿨다.
- [x] `style_tags`와 `style_concepts`를 분리하고, concept가 `style_tags`에 잘못 들어오면 후처리로 이동시키게 했다.
- [x] multi-item query parsing을 강화했다.
- [x] `with`, `and`, `&`, `plus`, `paired with`, `layered with`, `over`, `under`, `,` connector를 지원한다.
- [x] item별 `raw_phrase`를 보존하고, explicit color/material/silhouette/style_concepts를 phrase 기준으로 바인딩한다.
- [x] color prompt를 강화해서 visible wearable item의 color를 더 적극적으로 채우게 했다.
- [x] detail / evidence / canonical color 기반 color backfill을 추가했다.
- [x] outfit-level canonical color를 item 전체에 무작정 복사하지 않도록 유지했다.
- [x] multi-item full-set match 중심의 symbolic ranking을 적용했다.
- [x] cross-item swap penalty를 유지하고 강화했다.
- [x] hybrid rerank를 적용했다.
- [x] `final_ranking_mode = "hybrid_weighted"`로 기록한다.
- [x] `score_breakdown`에 symbolic / embedding / concept / detail 기반 hybrid score를 남긴다.
- [x] dense retrieval에 structured serialized query를 실제 반영한다.
- [x] 820개 실험 번들 기준 `merge -> backfill -> build-index` 산출물 파일을 생성했다.
- [x] 820개 query set 실행 및 labeling sheet 생성을 위한 `run-query-set` CLI를 붙였다.

### Changed From Original Plan
- [x] dense retrieval은 더 이상 `raw query + serialized query` embedding 평균을 쓰지 않는다.
- [x] 현재 dense retrieval은 `serialized query only`를 사용한다.
- [x] `late fusion`은 추후 실험 예정으로 메타데이터에 기록해 두었다.

### In Progress
- [ ] 820개 실험 번들 산출물 품질 확인 및 sign-off
- [ ] 820개 기준 query regression 확인
- [ ] `mood-heavy / multi-item binding / color-sensitive` query set을 고정하고 labeling/eval 입력 자산으로 정리하기
- [ ] 공식 metric 계산 루틴을 붙여 `Top-1 exact`, `Top-5 acceptable`, `item binding accuracy`, `color contradiction rate`, `concept miss rate`를 자동 기록하기
- [ ] 820개 beyond full archive로 태깅 범위 확장

### Not Yet Closed
- [ ] `serialized_only`와 향후 `late fusion`을 동일 query set에서 비교하기
- [ ] 전체 archive 기준 운영 파이프라인 정리

### Immediate Next Work
1. 820개 기준 고정 query set을 먼저 확정한다.
2. `run-query-set` 산출물을 기준으로 regression / labeling을 한 번 닫는다.
3. 공식 metric 계산 루틴을 붙여서 query set 결과를 수치로 남긴다.
4. 여기까지 안정화되면 full archive 확장으로 넘어간다.

### Later
1. `serialized_only` vs `late fusion` 비교 실험
2. full archive 운영 파이프라인 문서화
3. 필요 시 parser의 rule-first category coverage를 footwear/accessory까지 확장

### Notes
- 현재 구현은 문서의 기존 설명보다 더 precision-first 쪽으로 조정되어 있다.
- 특히 dense retrieval은 지금 `serialized_only` baseline으로 고정되어 있고, `late fusion`은 아직 구현하지 않았다.
- `merge -> backfill -> build-index` 산출물 자체는 이미 존재하지만, regression/eval sign-off는 아직 끝나지 않았다.
- `run-query-set`은 query preview + labeling sheet 생성용이고, Pinterest-like 공식 metric 집계까지는 아직 없다.
- 따라서 아래 계획 본문 중 dense retrieval 설명은 일부 최신 구현과 다를 수 있다.



# V3 Precision Upgrade for `mood + multi-item + color`

## Summary
V3를 Pinterest처럼 "쿼리 의도에 딱 맞는 이미지" 중심으로 바꾸려면, 이번 수정은 랭킹만 만지는 방식으로는 부족합니다.

1. `query 이해`를 item-bound 구조로 강화한다.
2. `document/item tagging`을 다시 만들어 mood와 color를 item/룩 수준에서 더 정확히 보강한다.
3. `retrieval + rerank`가 raw query 문자열이 아니라 구조화된 intent를 실제로 사용하게 바꾼다.

이번 버전에서는 `vintage`를 다른 style concept와 동일하게 취급합니다.  
즉 `vintage`만 별도 hard rule로 승격하지 않고, `minimal`, `romantic`, `avant-garde`와 같은 동일한 concept layer 안에서 처리합니다.

## Implementation Changes

### 1. Query schema를 item-bound + concept-aware로 강화
- `V3TargetItem`에 `style_concepts: list[str]`를 추가한다.
- `style_tags`는 `peep-toe`, `double-breasted`, `cropped` 같은 item descriptor로 유지한다.
- `style_concepts`는 `vintage`, `minimal`, `romantic`, `avant-garde`, `retro` 같은 검색용 concept만 담는다.
- `vintage`는 다른 concept와 동일하게 처리한다.
- explicit raw phrase에 concept가 있으면 기본적으로 `preferred`, 사용자가 강하게 제한하는 문맥일 때만 `required`로 승격한다.
- multi-item parser fallback을 rule-assisted로 보강한다.
- 분할 기준 connector는 최소 `with`, `and`, `&`, `plus`, `paired with`, `layered with`, `over`, `under`, `,` 를 지원한다.
- 각 item은 `raw_phrase`를 보존하고 explicit color/material/silhouette/style_concepts는 그 phrase 기준으로만 바인딩한다.
- dense retrieval용 `query serialization`을 추가한다.
- dense query는 현재 `serialized query only`를 encode한다.

### 2. Mood/style concept layer 추가
- item 쪽에 `style_concepts: list[str]`를 추가하고, document-level concept support는 item aggregation/inference로 처리한다.
- `style_concepts`는 아래 source에서 만든다.
  - item extraction output
  - canonical `mood`
  - canonical `era`
  - raw `mood`
  - raw `era`
  - `detail`
- 1차 concept catalog는 코드 상수로 고정한다.
- 최소 매핑:
  - `vintage`: `vintage`, `retro`, `worn-in`, `washed`, `distressed`, `aged`, `70s`, `1970s`, `80s`, `1980s`, `90s`, `1990s`, `2000s revival`
  - `minimal`: `minimal`, `clean`, `pared-back`
  - `romantic`: `romantic`, `soft`, `delicate`
  - `avant-garde`: `avant garde`, `avant-garde`
- 룩-level concept는 주요 apparel item에 backfill한다.
- accessory-only 전파는 하지 않는다.
- item extractor prompt를 수정해 `style_concepts`를 직접 추출하게 한다.
- `style_tags`와 `style_concepts`를 혼합하지 않는다.
- 820개 실험 번들은 `image_assisted`로 재태깅했고, `style_concepts backfill` 및 upgraded 산출물 생성까지 완료했다.
- 남은 일은 820개 기준 regression/eval sign-off다.

### 3. Color tagging을 strict하게 재설계
- item extractor prompt에서 visible wearable item의 color를 최대한 채우게 한다.
- `color`는 dominant color 기준 최대 2개만 허용한다.
- `two-tone`, `multicolor`, `color-block`는 color 외에 별도 concept/style tag로도 남긴다.
- color backfill rule을 추가한다.
- backfill 조건:
  - item color가 비어 있음
  - detail phrase 안에 해당 category 앞쪽 color token이 존재
  - 또는 단일-apparel look이고 canonical color가 단일값임
- outfit-level canonical color를 multi-item look에 무조건 복사하지 않는다.
- ranker는 explicit color query가 있을 때 strict하게 유지한다.
- color missing은 `contradiction`보다 `missing`으로 처리하고 dense recall 후보에는 남긴다.
- final top ranking에서는 `required color missing`은 계속 block한다.

### 4. Multi-item matching을 set match로 고정
- multi-item query는 item 수만큼 모두 충족돼야 top result 후보가 된다.
- same-category repeated item query도 지원한다.
- cross-item swap penalty는 유지하되 강화한다.
- `full item set match`가 아니면 top-1에 오르기 어렵게 조정한다.
- dense retrieval candidate 생성도 item count awareness를 반영한다.
- serialized query에 item order를 명시한다.

### 5. Retriever와 ranker를 실제 hybrid로 변경
- dense retrieval은 raw query 문자열만 쓰지 않고 serialized query를 반영한다.
- final ranking feature에 `embedding_support_score`를 추가한다.
- 기본 가중치:
  - `0.60 * normalized_symbolic_item_score`
  - `0.25 * normalized_embedding_score`
  - `0.10 * concept_support_score`
  - `0.05 * detail_consistency_score`
- `concept_support_score`는 `style_concepts` exact/partial 기준으로 계산한다.
- `vintage`도 다른 concept와 같은 방식으로 점수화한다.
- explicit color query는 embedding score가 높아도 color 위반이면 top 결과에서 제외한다.
- `retrieval_metadata["final_ranking_mode"]`는 `hybrid_weighted`로 변경한다.
- `late fusion`은 아직 미구현이고 메타데이터에 future work로만 남겨 둔다.

### 6. Offline eval을 Pinterest-like 기준으로 재구성
- query set 실행 및 labeling sheet 생성을 위한 CLI는 있다.
- 다만 공식 metric 계산 루틴은 아직 없다.
- query set을 최소 3축으로 관리한다.
  - `mood-heavy`
  - `multi-item binding`
  - `color-sensitive`
- 대표 쿼리 예시:
  - `vintage loose pants`
  - `white trousers with vintage black jacket`
  - `black trousers`
  - `ivory dress`
  - `red bag with beige coat`
  - `minimal black dress`
- 측정 기준:
  - `Top-1 exact intent satisfaction`
  - `Top-5 acceptable results`
  - `item binding accuracy`
  - `color contradiction rate`
  - `concept miss rate`

## Public API / Type Changes
- `V3TargetItem`
  - 추가: `style_concepts: list[str]`
- `V3DocumentItem`
  - 추가: `style_concepts: list[str]`
- `V3RankedResult.score_breakdown`
  - 추가: `embedding_support:*`, `concept_support:*`, normalized hybrid score
- `V3PipelineOutput.retrieval_metadata`
  - `final_ranking_mode = "hybrid_weighted"`

기존 `style_tags`는 유지하되 검색 핵심은 `style_concepts`로 옮긴다.

## Test Plan
- parser
  - `vintage black jacket and white trousers`가 2 item으로 분리되는지
  - `and`, `&`, `plus`, `layered with` 지원
  - explicit concept가 `style_concepts`에 들어가는지
- preprocessing / extractor
  - `era/mood/detail`에서 `style_concepts` backfill 되는지
  - color missing item이 detail phrase로 backfill 되는지
- ranker
  - multi-item full set match가 partial보다 항상 우선하는지
  - explicit color contradiction가 embedding high score보다 우선 차단되는지
  - `vintage` query가 `minimal` query와 동일한 concept scoring 규칙을 타는지
- retriever
  - dense query가 serialized query를 반영하는지
  - union mode에서 dense recall 후보가 hybrid rerank에서 적절히 승격되는지
- integration
  - 820-doc 실험 번들 재태깅 후 query set 회귀 테스트
  - 기존 single-item black/white trouser 테스트가 깨지지 않는지

## Assumptions / Defaults
- 이번 변경은 먼저 `820개 실험 번들`에 적용하고, 안정화 후 전체 archive로 확장한다.
- 재태깅은 `image_assisted`를 기본값으로 사용한다.
- `Pinterest-like`는 broad semantic match보다 query intent exactness를 우선하는 것으로 해석한다.
- `vintage`는 별도 예외가 아니라 일반 style concept 규칙을 따른다.
