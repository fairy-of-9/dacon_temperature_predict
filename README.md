# dacon_temperature_predict
DACON AI프렌즈 시즌1 온도 추정 경진대회

### 학습

- train, dev set
  - Y0~17만 가지는 경우를 train set으로 하고 해당 평균값을 label로 사용.
  - Y18이 주어진 경우는 dev set으로 사용.
  - train set으로 학습 후 dev set에서의 점수가 가장 높은 epoch 모델을 선택.
- features
  - 기온, 현지기압, 풍속 등 8 종류의 features를 학습.
    - feature마다 5개의 연속적인 값을 가짐.
  - feature 당 각각의 Bi-LSTM으로 학습.
  - 8개 LSTM의 output을 concat 후 Linear 연산으로 1개의 값으로.
- MSELoss 사용.

