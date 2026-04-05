### V1 pipeline
[[Image_Module]]
  

## 목적
이 문서는 `personal/image_module_v1.md`에 새로 정리한 V1 정의를 기준으로, `이미지 기반 1차 retrieval`만 먼저 검증하기 위한 실행 계획서다.


현재 V1은 아래 흐름으로 해석한다.

1. user text input

2. balance score 기반으로 생성 이미지 개수 조절

3. 로컬 CLIP류 오픈소스 모델로 archive 이미지 embedding

4. query text input + 생성된 AI 이미지를 기준으로 archive 1차 top-k 반환

즉 이번 문서의 범위는 `태그 기반 text retrieval`이 아니라 `멀티모달 이미지 retrieval baseline`이다.


## 이 설계에 대한 평가

이 방향은 전체적으로 좋다. 특히 아래 세 가지가 좋다.

1. `V1 = image retrieval`, `V2 = text/tag retrieval`로 역할이 분리되어 실험 해석이 쉬워졌다.

2. archive 전체를 미리 embedding 해두고 online에서는 query만 처리하는 구조라 반복 실험에 유리하다.

3. 나중에 V3에서 V1, V2를 hybrid로 합칠 때 각 축의 기여도를 따로 볼 수 있다.

다만 지금 단계에서 주의할 점도 분명하다.


1. V1의 품질은 생성 이미지 품질에 크게 흔들릴 수 있다.

2. 1차 retrieval이 약하면 tag rerank 이전에 embedding model, query fusion, generation prompt를 먼저 봐야 한다.

  

결론적으로:

  

- 구조 분리는 적절하다

- V1을 먼저 단독 검증하는 것도 맞다

  
  

## 현 트렌드 대비 평가

  

현재 이미지 추천 / 멀티모달 retrieval의 큰 흐름과 비교하면, 지금 V1은 `낡은 방식`이라기보다 `좋은 baseline`에 가깝다.

  

맞닿아 있는 지점:

  

1. shared embedding space에서 retrieval 하는 구조 자체는 여전히 정석이다

2. image branch와 text branch를 함께 쓰는 것도 현 추세와 맞는다

3. 최종적으로 rerank를 붙이는 구조도 실무적으로 매우 일반적이다

  

차이가 나는 지점:

  

1. 최신 방법은 단순 weighted cosine에서 끝나지 않고 learned combiner를 쓰는 경우가 많다

2. 패션 retrieval 쪽은 fashion-specific encoder나 reranker를 추가하는 경우가 많다

3. `generated image`를 query proxy로 쓰는 건 흥미로운 실험축이지만, 최신 benchmark 표준 포맷의 중심은 아니다

  

정리:

  

- 지금 V1은 baseline으로 적절하다

- V2와 결합하는 방향도 적절하다

- 다만 SOTA에 더 가까워지려면 다음 단계에서 `query composition`과 `reranking`을 학습형으로 확장해야 한다

  

## 핵심 설계 결정

  

### 1. archive embedding은 local model로 간다

  

현재 V1 목적에는 `CLIP / OpenCLIP / SigLIP` 계열이 맞다.

  

이유:

  

- archive는 이미지

- query는 text + 생성 이미지

- 같은 shared embedding space에서 비교해야 한다

  

OpenAI의 `text-embedding-3-large`는 text 전용이라 V1에는 맞지 않는다.

  

### 2. query는 text와 generated image를 둘 다 사용한다

  
V1 정의에 따르면 query속 image와 text의 값을 합하는 방식은 아래와 같다

  

- user query text

- generated reference image 1~4장

  

즉 V1은 "이미지 only"가 아니라 `multimodal query -> image archive search`다.

  

### 3. query속 image-text 값 결합 방식은 score-level fusion으로 고정한다

  

CLIP shared space에서는 text vector와 image vector를 직접 평균해도 되지만, 첫 실험은 score-level fusion이 더 해석 가능하다.

   아래 식으로 통일한다.

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

생성 이미지가 여러 장이면 먼저 하나의 `generated_image_vec`로 합친다.

  

```text

generated_image_vec = average(gen_i_vecs)

```

  

초기 가중치 추천:

  

- `w_text = 0.35`

- `w_image = 0.65`

  

이유:

  

- text는 의도 보정 역할

- generated image는 visual anchor 역할

  

즉 여러 장을 만들더라도 최종 retrieval 단계에서는 항상 아래 식만 쓴다.

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

이렇게 두면 나중에 결과가 안 좋을 때 어느 축이 문제인지 보기가 쉽다.

  

## 현재 데이터 기준

  

- 데이터 루트: `/Users/nojonghyeon/Documents/GitHub/For_switch_query_v2/data/2026/spring-ready-to-wear`

- 전체 이미지 수: `11,299`

- 확장자 분포: `11,298 jpg`, `1 jpeg`

- 브랜드 수: `365`

- 폴더 구조: `data/2026/spring-ready-to-wear/{brand}/collection/{image}.jpg`

  

즉 1차 retrieval을 위해서는 이 `11,299`장을 한 번 전부 embedding 해야 한다.

  

## 실험 목표

  

### Goal 1. 1차 retrieval만으로 후보군이 괜찮은지 확인

  

- top-k가 시각적으로 충분히 관련성이 있는지 본다

- text/tag rerank 없이도 쓸 만한 후보군이 모이는지 본다

  

### Goal 2. query text와 generated image의 역할 확인

  

- text만 썼을 때보다 generated image가 실제로 retrieval을 개선하는지 본다

- fusion weight를 어떻게 잡아야 하는지 감을 잡는다

  

### Goal 3. V2 필요성 판단

  

- V1만으로 후보군 recall이 괜찮으면 V2는 rerank 역할

- V1이 약하면 V2 이전에 V1 자체를 재설계해야 한다

  

## 실행 설계

  

### Step 0. inventory 생성

  

기존 함수 활용:

  

- `switch_query.image_module.preprocessing.build_image_inventory()`

  

예상 산출물:

  

- `tmp/sr26_inventory.csv`

  

필수 컬럼:

  

- `image_id`

- `file_path`

- `brand`

- `year`

- `season_group`

- `source_type`

- `filename`

  

### Step 1. archive 전체 image embedding 생성

  

입력:

  

- archive 이미지 전체 `11,299`장

  

방법:

  

- local CLIP류 오픈소스 모델로 각 이미지를 embedding

  

권장 시작점:

  

- `OpenCLIP ViT-B/32` 또는 비슷한 소형 계열

  

이유:

  

- M2 Air 8GB 환경에서 가장 무난한 출발점

- 품질과 자원 사용량 사이 균형이 괜찮음

  

저장:

  

- `image_id -> vector`

- `image_id -> file_path / brand / metadata`

  

예상 산출물:

  

- `tmp/sr26_v1_archive_vectors.npy`

- `tmp/sr26_v1_archive_meta.csv`

  

권장:

  

- vector는 `numpy` binary 저장

- metadata는 csv/json 저장

- 검색은 처음엔 단순 cosine search로 시작

- 필요하면 이후 `faiss`로 전환

  

이 단계는 offline 1회성 작업이다.

  

### Step 2. query별 생성 이미지 생성

  

입력:

  

- user query text

- balance score

  

출력:

  

- reference image 1장 또는 3~4장

  

정책:

  

- 수렴 query: `1장`

- 발산 query: `3~4장`

  

예상 산출물:

  

- `tmp/generated_refs/{query_id}_1.png`

- `tmp/generated_refs/{query_id}_2.png`

  

모델 후보:

  

- `gpt-image-1-mini`

- `gpt-image-1.5`

  

첫 실험 추천:

  

- 비용을 줄이려면 `gpt-image-1-mini`

- 이미지 퀄리티를 보려면 `gpt-image-1.5`

  

### Step 3. query text embedding + generated image embedding 생성

  

입력:

  

- query text

- generated reference image 1~4장

  

방법:

  

- archive와 동일한 CLIP 모델의 text encoder로 query text embedding

- archive와 동일한 CLIP 모델의 image encoder로 generated image embedding

  

출력:

  

- `query_text_vec`

- `generated_image_vec`

  

예상 저장:

  

- `tmp/sr26_v1_query_vectors.json`

  

생성 이미지가 여러 장이면:

  

```text

generated_image_vec = average(gen_i_vecs)

```

  

주의:

  

- text encoder와 image encoder가 반드시 같은 model family여야 함

- 그래야 shared embedding space 의미가 유지된다

  

### Step 4. 1차 similarity search

  

archive 각 이미지에 대해 아래 점수를 계산한다.

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

초기 기본값:

  

- `w_text = 0.35`

- `w_image = 0.65`

  

출력:

  

- top-k 후보

  

기본값:

  

- `k = 20`

  

예상 산출물:

  

- `tmp/sr26_v1_first_stage_topk.csv`

  

필수 컬럼:

  

- `query_id`

- `query_text`

- `rank`

- `image_id`

- `final_score`

- `text_score`

- `image_score`

- `brand`

- `file_path`

  

여기서 보조 로그용 점수는 아래처럼 저장한다.

  

```text

text_score = cosine(query_text_vec, archive_image_vec)

image_score = cosine(generated_image_vec, archive_image_vec)

```

  

이렇게 저장해야 나중에 `text가 문제인지`, `generated image가 문제인지` 해석 가능하다.

  

### Step 5. 정성 평가

  

각 query에 대해 아래를 확인한다.

  

1. top-k가 시각적으로 비슷한가

2. color / silhouette / mood 수준에서 유사성이 보이는가

3. generated image를 넣었을 때 text-only보다 결과가 좋아지는가

4. 특정 브랜드/쇼만 반복적으로 과도하게 뜨는가

  

이번 단계의 성공 기준은 "최종 추천 완성"이 아니라 "rerank 전 후보군 recall이 볼 만한가"다.

  

## 비교 실험 권장안

  

V1은 아래 3개 모드로 꼭 비교하는 것이 좋다.

  

1. `text-only`

  

```text

final_score = cosine(query_text_vec, archive_image_vec)

```

  

2. `generated-image-only`

  

```text

final_score = cosine(generated_image_vec, archive_image_vec)

```

  

3. `fusion`

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

이 3개를 같이 보면 V1 내부에서도 무엇이 실제로 기여하는지 바로 알 수 있다.

  

## SOTA-근접 로드맵

  

현재 문서의 V1을 유지하면서, 이후 어떤 순서로 업그레이드하면 좋은지 정리한다.

  

### Stage A. 현재 V1 baseline 고정

  

목표:

  

- `text-only`

- `generated-image-only`

- `fusion`

  

이 세 모드를 먼저 비교해서 어떤 입력 축이 실제로 기여하는지 확인한다.

  

이 단계는 "작동하는 baseline 확보"가 목적이다.

  

### Stage B. backbone 개선

  

현재 V1에서 가장 먼저 바꿔볼 만한 것은 scoring 식보다 backbone이다.

  

우선순위:

  

1. `OpenCLIP ViT-B/32` baseline

2. 가능하면 더 강한 `SigLIP` 또는 `SigLIP2` 계열 비교

  

이 단계의 질문:

  

- 같은 scoring 식이어도 backbone만 바꾸면 top-k 품질이 개선되는가

  

### Stage C. learned query composition

  

현재 V1은 아래처럼 고정 가중치 결합이다.

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

SOTA에 더 가까운 방향은 이 가중치 결합 대신, `query text + generated image`를 받아 하나의 retrieval query vector를 학습적으로 만드는 것이다.

  

즉 다음 단계에서는:

  

- weighted cosine baseline

- learned combiner / adapter

  

를 비교하는 방향으로 갈 수 있다.

  

### Stage D. fashion-specific rerank

  

패션 retrieval에서는 상위 후보군이 어느 정도 맞더라도, 세부 속성에서 흔들릴 수 있다.

  

그래서 2차 단계에서는 아래가 유효하다.

  

- tag-based rerank

- fashion attribute rerank

- cross-encoder style rerank

  

이 단계가 현재 문서의 V2, 이후 V3와 자연스럽게 이어진다.

  

### Stage E. hybrid final system

  

최종적으로는 아래 구조가 가장 현실적이다.

  

1. V1으로 broad visual recall 확보

2. V2로 semantic/attribute precision 보정

3. V3에서 hybrid score로 최종 top-k 확정

  

즉 현재 V1은 버릴 실험이 아니라, 최종 시스템의 recall stage가 될 가능성이 높다.

  

## 추천 실험 query

  

초기 점검용:

  

1. `black tailored coat`

2. `romantic white dress`

3. `sporty minimal look`

4. `sheer layered look`

5. `red floral dress`

6. `structured jacket with sharp shoulders`

  

이 query들은 visual retrieval 관점에서 결과 차이를 보기 좋다.

  

## M2 Air 8GB 기준 운영 가이드

  

이 환경에서는 V1 프로토타입은 가능하지만, 크게 무리하면 답답해질 수 있다.

  

권장 운영 방식:

  

1. 작은 CLIP 계열로 시작

2. batch size는 `1~8` 수준으로 작게 시작

3. 먼저 `500~1000장 subset`으로 검증

4. 괜찮으면 전체 `11,299장` embedding은 offline으로 1회 실행

5. 벡터 캐시를 만든 뒤에는 재사용

  

즉 이 머신에서 중요한 건 "매 쿼리마다 전부 다시 계산하지 않는 구조"다.

  

## 기존 내용에서 바뀐 점

  

이번 문서에서 기존 내용 대비 바뀐 핵심은 아래와 같다.

  

1. `V1을 단순 image-only retrieval`로 적지 않고, `query text + generated image`를 함께 쓰는 멀티모달 retrieval로 명확히 정의했다.

2. 4번 결합 방식을 추상적으로 두지 않고 아래 식으로 고정했다.

  

```text

final_score

= w_text * cosine(query_text_vec, archive_image_vec)

+ w_image * cosine(generated_image_vec, archive_image_vec)

```

  

3. 생성 이미지가 여러 장일 때는 retrieval 단계 전에 평균 벡터로 합친다는 규칙을 추가했다.

4. `text-only / generated-image-only / fusion` 3모드 비교를 실험 기본값으로 추가했다.

5. M2 Air 8GB 환경을 고려해 small backbone, small batch, subset-first 전략을 명시했다.

6. 현재 방식이 트렌드에서 완전히 벗어난 것이 아니라, `좋은 baseline`이며 이후 learned combiner와 rerank로 확장 가능한 구조라는 평가를 추가했다.

  




## 보류 항목

  

- V2 text/tag retrieval

- canonical tag normalization

- 2차 rerank

- V1/V2 hybrid score tuning

  

## 현재 완료.

  

1. V1용 local CLIP model을 하나 확정

2. archive embedding 스크립트 작성

3. query generation 스크립트 작성

4. text/image fusion retrieval 스크립트 작성

5. `text-only`, `image-only`, `fusion` 3모드 비교

6. 결과 확인 후 V2 rerank 설계