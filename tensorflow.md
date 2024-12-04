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
장