### Shape
#### - Tensor(텐서)의 차원 및 크기를 나타내는 속성. 서는 다차원 배열이므로, 각 차원의 데이터 크기를 Shape로 정의함
- 구성요소
  - Rank(차원): 텐서가 몇 개의 축(axis)을 가지는지 나타낸다.
  - ex> 스칼라(0D), 벡터(1D), 행렬(2D), 다차원 텐서(3D 이상)
- 각 차원의 크기
  - 각 축의 데이터 개수를 나타낸다.
  - ex>
    [2, 3]은 2행 3열의 2D 텐서

```python
import tensorflow as tf

# 텐서 정의
scalar = tf.constant(5)                 # 스칼라 (Rank 0)
vector = tf.constant([1, 2, 3])         # 벡터 (Rank 1)
matrix = tf.constant([[1, 2], [3, 4]])  # 행렬 (Rank 2)

# Shape 확인
print("Scalar Shape:", scalar.shape)   # ()
print("Vector Shape:", vector.shape)   # (3,)
print("Matrix Shape:", matrix.shape)   # (2, 2)
```

### constant와 Variable의 정의 및 차이점

#### **`tf.constant`**
- **정의**: 변경 불가능한 값(Immutable)을 가진 상수 텐서를 생성
- **용도**: 입력 데이터나 변경되지 않는 매개변수 정의
- **특징**: 한 번 생성된 텐서는 값을 변경할 수 없음

#### **`tf.Variable`**
- **정의**: 변경 가능한 값(Mutable)을 가진 학습 가능한 텐서를 생성
- **용도**: 딥러닝 모델의 가중치(weight)와 바이어스(bias) 정의
- **특징**: 값 변경 가능 (`assign`, `assign_add` 등 메서드 사용)

#### **차이점**

| **특징**               | **`tf.constant`**                              | **`tf.Variable`**                              |
|------------------------|-----------------------------------------------|-----------------------------------------------|
| **값 변경 가능 여부**    | ✖ 불가능                                       | ✔ 가능                                         |
| **용도**               | 변경할 필요가 없는 값                          | 학습 가능한 파라미터 (가중치, 바이어스)        |
| **초기화 필요 여부**     | 필요 없음                                      | 초기값 필수                                     |
| **예제**               | `tf.constant([1.0, 2.0])`                     | `tf.Variable([1.0, 2.0])`                     |

#### **코드 예제**

```python
import tensorflow as tf

# tf.constant
const_tensor = tf.constant([1.0, 2.0, 3.0], name='constant_tensor')
print("Constant Tensor:", const_tensor)

# tf.Variable
var_tensor = tf.Variable([1.0, 2.0, 3.0], name='variable_tensor')
print("Variable Tensor (before):", var_tensor)

# tf.Variable 값 변경
var_tensor.assign([4.0, 5.0, 6.0])  # 값 변경
print("Variable Tensor (after):", var_tensor)
```

### 수식과 TensorFlow 코드 매칭

| **수식 구성 요소**               | **TensorFlow 코드**                                   | **설명**                                                                                   |
|----------------------------------|------------------------------------------------------|------------------------------------------------------------------------------------------|
|  `h(x_i) - y_i`               | `hypothesis - y_train`                               | 예측값과 실제값의 차이를 계산.                                                            |
| `(h(x_i) - y_i)^2`           | `tf.square(hypothesis - y_train)`                   | 차이를 제곱. <br> **`tf.square`: 텐서의 각 요소를 제곱.**                                   |
| `Σ` (합산)             | `tf.reduce_sum(tf.square(hypothesis - y_train))`    | 제곱된 값들을 모두 합산. <br> **`tf.reduce_sum`: 텐서의 모든 값(또는 지정 축(axis)의 값)을 합산.** |
| `(1/m) Σ` (평균 계산)   | `tf.reduce_mean(tf.square(hypothesis - y_train))`   | 합산된 값을 데이터 개수 \( m \)로 나누어 평균을 계산. <br> **`tf.reduce_mean`: 평균값 계산.**      |

---

1. **`tf.square`**:
  - 텐서의 각 요소를 제곱
  - 입력: `tf.Tensor([2, -3])` → 출력: `tf.Tensor([4, 9])`

2. **`tf.reduce_sum`**:
  - 텐서의 모든 요소(또는 축 지정 시 해당 축의 값)를 합산
  - 입력: `tf.Tensor([4, 9])` → 출력: `13` (합계)

3. **`tf.reduce_mean`**:
  - 텐서의 모든 요소(또는 축 지정 시 해당 축의 값)를 평균 계산
  - 입력: `tf.Tensor([4, 9])` → 출력: `6.5` (평균)

### lab03 & lab04
**`tf.random.set_seed`**:
- 난수 생성기의 시드를 설정하여 동일한 코드 실행 시 일관된 난수 값을 생성. 재현 가능성을보장하기 위해 사용됨

``` python
  tf.random.set_seed(777)
  print(tf.random.normal([1]))  # 항상 동일한 결과값 출력
```

**`tf.optimizers.SGD`**:
- Stochastic Gradient Descent (SGD) 옵티마이저를 생성. learning rate 를 입력받아 모델의 파라미터를 업데이트하는 데 사용

``` python
  optimizer = tf.optimizers.SGD(learning_rate=0.01) # SGD 옵티마이저 인스턴스 생성
```

**`tf.GradientTape`**:
- 자동 미분을 위해 사용되는 클래스. 연산의 모든 과정을 기록하여, 이후 특정 변수에 대한 기울기(gradient)를 계산

``` python
  x = tf.Variable(3.0)
  with tf.GradientTape() as tape:
      y = x**2
  grad = tape.gradient(y, x)  # dy/dx = 2 * x
  print(grad)  # 출력: tf.Tensor(6.0, shape=(), dtype=float32)
```
**`zip`**:
- Python 내장 함수로, 여러 iterable 객체를 병렬 처리하기 위해 묶음
- 입력: `zip([1, 2], ['a', 'b'])` → 결과: 병렬로 `(1, 'a')와 (2, 'b')` 형태의 튜플 생성
- TensorFlow 에서 사용: `zip(gradients, variables)` 로 기울기와 변수 쌍을 묶어 `apply_gradients` 에 전달  

``` python
  gradients = [0.1, 0.2]
  variables = ['w1', 'w2']
  for g, v in zip(gradients, variables):
      print(g, v)  # 출력: 0.1 w1 / 0.2 w2
```
**`optimizer.apply_gradients`**:
- 계산된 기울기를 기반으로 파라미터를 업데이트
- 입력: `optimizer.apply_gradients(zip(gradients, variables))`
→ 결과: 변수들이 기울기를 반영하여 업데이트

``` python
  gradients = [tf.constant(0.1), tf.constant(0.2)]
  variables = [tf.Variable(1.0), tf.Variable(2.0)]
  optimizer = tf.optimizers.SGD(learning_rate=0.1)
  optimizer.apply_gradients(zip(gradients, variables))
  print(variables[0].numpy())  # 결과: 0.99 (1.0 - 0.1 * 0.1)
```