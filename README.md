# 한성대학교 캡스톤 결과물 중 하나 입니다.
## 프로젝트 전체 git 주소는
https://github.com/Mshhhhh/CFAD
입니다.

## 1. 프로젝트 수행 목적

### 1.1 프로젝트 정의
Deep Learning을 이용하여 감정인식 기반의 칵테일 추천

### 1.2 프로젝트 배경
개인의 취향과 감정은 소비 결정 과정에서 중요한 역할을 한다. 칵테일 선택 시 다양한 맛과 색상이 있지만, 사용자의 실시간 감정 상태를 반영한 개인화된 칵테일 추천은 드물다. 현재 시장에서는 감정이나 취향을 고려하여 칵테일을 추천하는 서비스가 부족하기 때문에, 이를 제공하는 웹사이트를 제작하게 되었다.

### 1.3 프로젝트 목표
- 감정인식 AI 모델을 구축하고 영상 분석을 이용해 사용자의 감정 상태를 인식
- Deep Learning을 활용한 감정인식 결과와 선호도 조사 데이터를 가중치와 알고리즘을 통해 분석하여, 개인 상태에 가장 적합한 칵테일 추천 기능 구현

---

## 2. 프로젝트 결과물 개요

### 2.1 프로젝트 구조
웹사이트 기반 감정인식 칵테일 추천 시스템

![image](https://github.com/user-attachments/assets/48156a85-d11a-48c8-aa5f-66225561ffcd)

### 2.2 프로젝트 결과물
- **시작화면**
  
![image](https://github.com/user-attachments/assets/0b99df02-b333-44b8-8909-75f18bb656e7)

- **메인 페이지**: 사용자의 이름 입력
  
![image](https://github.com/user-attachments/assets/d9044af4-4bb9-4174-8c28-6d8663b9878b)

- **감정인식 및 선호도 조사**: AI 기반 감정 조사 또는 직접 선호도 입력
  
![image](https://github.com/user-attachments/assets/ca164dc8-4c20-4df7-bec5-38db0f841609)
- 웹캠이나 외부 카메라에 연결하여 유저의 얼굴 사진을 입력받을 수 있도록 함
  
![image](https://github.com/user-attachments/assets/56a5d962-189b-4c8c-a895-5e9aad0f71fa)

- **칵테일 추천**: 입력받은 데이터를 바탕으로 최적의 칵테일 추천
  
![image](https://github.com/user-attachments/assets/07fe09ca-12cd-4edd-beaa-7bbbd1b19bb5)

- **세부 정보 제공**: 칵테일 위에 커서를 올리면 도수, 베이스 등의 정보 표시
  
![image](https://github.com/user-attachments/assets/1fc58826-a843-487c-bfc6-3c961109b6fb)

---

## 3. 프로젝트 수행 추진 체계 및 일정

### 3.1 각 조직원의 조직도
![image](https://github.com/user-attachments/assets/658a5e60-358b-49f6-b057-b555ee0a34a5)

- **AI, API 개발**: 김종준
- **백엔드 개발**: 송형원
- **프런트엔드 개발**: 문서현, 송승원, 현상훈

### 3.2 역할 분담
- AI 모델 개발 및 API 구현
- 백엔드 서버 구축
- 프런트엔드 UI/UX 개발

### 3.3 주 단위 수행 일정
| 기간 | 작업 내용 |
|------|----------|
| 3월 | 계획 수립 및 자료 수집 |
| 4월 | 자료 분석 및 설계, 자문 및 수정 |
| 5월 | 서비스 제작 |
| 6월 | 최종 보고서 작성 |

---

## 4. 참고 자료
- AI데이터 허브
- OpneCV
- deepface
- Pytorch
- Wikipedia
- Google Image
- Open AI
- Animista


# 한성대학교 캡스톤 CFAD AI 파트

- 해당 파일 얼굴 데이터 경우 한국인 감정인식을 위한 복합 영상인
https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=82
- AI hub에서 데이터를 다운로드 받아서 데이터셋을 활용했습니다. 


## 데이터셋 재구축
- AI Hub에서 다운로드한 데이터가 40만 개를 넘지만, 해당 사진에 문제가 있음 발견.
- 전처리를 위해 기본적인 `.json`에 담겨있는 box 좌표를 이용하려 했으나, 좌표가 제대로 찍히지 않는 문제 발생.
- 전처리를 위한 Face Detection 모델 후보군 선정:
  - **cvrib**: CPU/GPU 자원 소모가 적지만 정확성이 낮음.
  - **Mediapipe Face Detection**: 속도가 느리지만 정확성이 높으며, 90%의 정확도 기록.
  - **Human-Face-Crop-ONNX-TensorRT**: GPU 자원 소모가 많지만 얼굴 검출 정확도가 높음. 그러나 비얼굴 객체에서도 얼굴을 검출하는 경우가 많음.
- **Mediapipe 얼굴 검출 사용 결정**
  - Mediapipe 사용 결과: **16시간 소요**
    
  ![image](https://github.com/user-attachments/assets/ea9e8689-a0e0-4c89-9727-2ba6e4a3103d)

  - 그러나, Meta에서 개발한 **DeepFace**의 성능이 월등히 좋아 최종적으로 DeepFace 사용 결정.
  - Mediapipe는 컬러 사진에 적합했지만, DeepFace는 흑백·컬러 모두에서 우수한 성능을 보임.
  - 100% 정확도가 아니므로, 얼굴이 아닌 검출 결과를 제거하는 추가 작업 진행.
  - 추가적인 데이터 압축을 위해 모든 사진 흑백화, 그리고 검출된 얼굴 부분을 제외한 나머지 부분은 삭제

## 모델 선정
- CNN, DNN, ResNet-50을 Pre-train하여 비교한 결과, **CNN이 가장 적합**한 모델로 결정됨.
- Train Loss가 수렴하지 않는 문제 발생:
  - 회전된 사진을 삭제했으나 학습이 원활하지 않음.
  - 사람이 구분하기 어려운 사진을 제거한 후 학습 재진행.
- **60%의 정확도 기록 시작**

## 모델 학습
- 데이터셋을 아무리 줄여도 **200GB 이하로 줄이기 어려움**.
- Google Colab 유료 버전 사용을 제외하고 **로컬 GPU를 활용하여 모델 학습 진행**.
- 학습 속도가 예상보다 느려 **훈련 epoch를 낮추는 방법으로 진행**.

