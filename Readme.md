## Dockerfile
docker build -t tensorflow .

## 컨테이너 실행
docker run -it --rm --name tensorflow -v $(pwd):/app -w /app/scripts tensorflow python <파일명>

---
# 머신러닝 (ML - Machine Learning)
머신러닝은 **명시적으로 프로그래밍하지 않아도** 데이터를 통해 학습하고, 새로운 데이터에 대한 예측을 수행하는 인공지능 기술이다.
알고리즘이 데이터를 분석하여 패턴을 발견하고 이를 바탕으로 결정을 내릴 수 있다.


## 학습 방법
### 1. Supervised
- 입력 데이터와 정답(라벨)을 함께 제공하여 모델을 학습시키는 방법
- 모델이 입력 데이터에 대해 올바른 출력을 예측하도록 학습
- 스팸 메일 분류, 공부 시간 별 시험 점수 결과

### 2. Unsupervised
- 정답(라벨) 없이 입력 데이터만 제공하여 데이터를 학습하는 방법
- 데이터 내의 패턴이나 군집(클러스터)을 찾는다.
- 고객 세그먼테이션, 문서 주제 분류


## Supervised 학습 방법의 종류

### 1. Regression
- 연속적인 숫자 값을 예측하는 문제
- **수식:** h(x) = W * x + b


### 2. Binary Classification
- 두 가지 클래스 중 하나로 데이터를 분류하는 문제 (pass / non-pass)
- ex> 스팸 메일 여부 (스팸/스팸 아님), 질병 진단 (양성/음성)

### 3 Multilevel Classification
- 여러 가지 클래스 중 하나로 데이터를 분류하는 문제
- ex> 동물 분류 (고양이, 개, 새 등), 이미지 속 사물 인식

---
## Linear Regression

### 1. 정의
**독립 변수(x)** 와 **종속 변수(y)** 간의 **선형 관계**를 모델링하여 값을 예측하는 머신러닝 알고리즘  
입력 데이터에 대해 가장 적합한 선을 찾아 예측값을 계산한다.

### 2. 가설 (Hypothesis)
선형 관계를 수식으로 표현: h(x) = W * x + b

- **W**: 기울기 (Weight)
- **b**: 절편 (Bias)


### 3. 비용 함수 (Cost Function)
예측값과 실제값의 차이를 측정하기 위해 **평균제곱오차(Mean Squared Error, MSE)** 를 사용: J(W, b) = (1 / m) * Σ (h(x(i)) - y(i))^2

- **m**: 데이터 개수
- **h(x(i))**: i번째 예측값
- **y(i)**: i번째 실제값




---
## Reference
모두를 위한 딥러닝 강좌 시즌 1 - https://www.youtube.com/watch?v=BS6O0zOGX4E&list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm&index=1

머신 러닝의 세 가지 종류 - https://tensorflow.blog/ml-textbook/1-2-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D%EC%9D%98-%EC%84%B8-%EA%B0%80%EC%A7%80-%EC%A2%85%EB%A5%98/
