
[[현재 switch-query 예상 아키텍처]]
에서 image module에 대해서 디테일한 설계를 하는 파일이다. 

현재 생각중인 쿼리 기반 이미지 추천 알고리즘은 이렇다. 

##### V1 pipeline 
1. user text input
2. balance score에 기반하여 (발산 혹은 수렴 단계)에 따라서 user content기반 image생성 개수 조절
-> 발산 상태에서는 더 다양한 combination이 필요하기에 3~4장, 수렴 상태는 조금 더 수렴해야기에 1장으로 고정
3. 사용자 업로드 이미지와 synthetic reference를 text로 tagging 진행-> 현재 작업 중. 더 자세히는 데이터의 태그 속성을 전처리하고, 동의어를 하나로 통일하는 작업 진행중.
-> v1에서는 최대한 간단한 baseline을 구성하기 위해서 vision embedding을 사용하는 더 적합한 버전 전에 image에 대해서 tagging을 하고 그 tag에 기반한 open ai embedding space속 text embed vector를 반환한다.(v2에서 업데이트 예정)
4. text-embedding-3-large로 query/document embedding 생성
5. top k 반환
6. 고정 속성 기반 rerank
7. 생성 이미지와 아카이브 속 이미지 같이 반환


###### i/o phase
input은 query text, 유저가 올린 이미지, stage, balance score
output은 archive_result, generated_result

###### 고정 schema
category, silhouette, color, material, pattern, texture, mood, season, era, detail 이렇게 일단 구성

###### db for local
local에서 일단 확인을 위한 db이기에 최대한 가볍게 구성하였다. 
google sheet로 관리 가능하게 csv로 return. 
embedding vector들은 cache에 저장

###### 동의어 처리 방식
일단 canonical label을 미리 정규화 해두자. 각 feature에 대해서 미리 정규화 
사용자 표현과 태그를 같은 표현으로 매핑 진행
->조금 더 디테일하게 잡아보자.  미리 각 카테고리 마다 fine하게 정규화 처리를 해두지 말고, llm이 뱉어내는 표현들을 수집해서 같은 의미를 중복된 표현 기반으로 통일 시키는 건 어떨까?
[[동의어+전처리.md]]

##### 향 후 업그레이드 방안
현재는 multimodal 이 아닌 text embedding 구조이기에 v2에서는 image-text embedding으로 전환 예정


