
![[Pasted image 20260328182502.png]]

전체 frame work에 대해서 설명을 하면 
1) 사용자는 디자이너다
2) 이 디자니어는 initial input에 만들고자 하는 컨셉, 그리고 관련 이미지를 입력값으로 첨부한다

##### Main Concept Modeling
1) 1번째에는 initial input을 바탕으로 
Main Concept Modeling block에서 세부 카테고리를 조직한다
2) 그리고 i+1번째부터는 context classifier에서 제공하는 맥락까지 고려하여 기존 세부 카테고리를 업데이트한다. 
3) 사용자는 또한 자신이 직접 이 트리 구조를 변형할 수 있다
##### balance module
1) 그리고 이 block에서 카테고리에 대한 맥락을 넣어주면 balance module은 현재 상황이 사용자가 발산을 원하는지(아이디어의 확장) 아니면 일관성(조금 더 수렴할 수 있게) 를 원하는지를 확인하여 diversity인지 consistency인지를 확인한다.
2) i+1번째부터는 context classifier에서 보내주는 맥락관련 정보도 포함해서 balance를 정한다.
##### context classifier
1) 사용자와 나눈 대화를 기반으로 현재 사용자가 어느 stage인지를 추측하여 stage를 분류하는 블록이다.+ 그 stage 안에서 어느 정도의 발산, 확산을 원하는지를 balance module이 측정할 수 있도록 input 제공하는 블록이다. 블록e.g) 의상 분야에 대해서는 mood board인지, sketch stage인지를 확인하는 블록
##### image module
1) Balance module에서 주는 정보를 바탕으로 이 블록 내에서 두가지 작업을 같이 진행한다. 
2) 이미지 태깅 기반으로 현재 맥락에 관련있는 기존 아카이브 데이터에서 사진들과 함께 llm이 직접 제작한 디자인 관련 이미지를 생산하여 핀터레스트식으로 여러가지를 병렬로 여러개 제시한다. 
3) 이 블록은 결국 사용자가 직접적으로 보는 이미지에 관여하는 핵심 블록이기에 이 블록이 점점 더 좋은 성능으로 update되는 하네스 기반 루프를 조성한다. 

