---
title: AdaBoost, Gradient Boosting
date: 2025-07-28 22:00:00 +0900
categories: [Paper, Machine Learning]
tags: [Boosting, AdaBoost, Adaptive Boosting, Gradient Boosting]
math: true
---

**Paper**: 

- Greedy function approximation: A gradient boosting machine **(Friedman)** ([https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full))

- Stochastic Gradient Boosting **(Friedman)**  ([https://www.researchgate.net/publication/222573328_Stochastic_Gradient_Boosting](https://www.researchgate.net/publication/222573328_Stochastic_Gradient_Boosting))

- AdaBoost and the Super Bowl of Classifiers A Tutorial Introduction to Adaptive Boosting **(Rojas)** ([http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf](http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf))

**Lecture Notes:**

- A Gentle Introduction to Gradient Boosting (Cheng Li) ([http://www.chengli.io/tutorials/gradient_boosting.pdf](http://www.chengli.io/tutorials/gradient_boosting.pdf))
- [https://en.wikipedia.org/wiki/Gradient_boosting](https://en.wikipedia.org/wiki/Gradient_boosting)

<br/>

회사에서 자주 사용하고 있는 LightGBM에 대한 공부를 하다가 Gradient Boosting를 공부하면서 정리하기로 했다. 아이디어 자체는 1990년대 후반 ~ 2000년대 초반에 나왔고, 이후 널리 사용되었기 때문인지 강의자료가 많았다. 그래서 논문 그 자체보다는 내가 이해하기 쉬웠던 강의 자료들 위주로 정리한다.

Gradient Boosting을 이해하려면 Boosting을 이해해야 하는데, 항상 같이 나오고 비교되는게 AdaBoost이므로 이것도 같이 정리한다.

<br/>

## 1. Background

---

#### 1-1. Boosting 이란

boosting은 기본적으로 weak model 여러 개를 weighted sum을 해서 boosted model을 만드는 것을 의미한다. 이를 수식으로 표현하면 아래와 같다:


$$
H_T(x_i) = \sum_{t}^{T} \rho_t h_t(x_i)
$$


여기서 각 weak model `h`를 어떻게 학습할지, 그리고 각 weak model의 가중치 `ρ` 를 어떤 방식으로 계산할지에 따라 어떤 Boosting 기법인지가 갈린다.

<br/>

## 2. Theory

---

#### 2-1. AdaBoost (Adaptive Boosting)

##### 2-1-A. weak model h를 어떻게 학습하나?

Binary Classification을 염두한 **Rojas**는 t 단계에서 학습할 weak model `h_t`를 결정하기 위해 exponential loss를 error로 설정하여 아래와 같이 유도한다:


$$
\begin{align}
E_t(X) 
&= \sum_{i} e^{-y_i H_t(x_i)} = \sum_{i}e^{-y_i H_{t-1}(x_i)} \cdot e^{-y_i \rho_t h_t(x_i)}
\\
&\equiv \sum_{i} w_{i}^{(t)} \cdot e^{-y_i \rho_t h_t(x_i)}
\\
&= 
\sum_{y_i h_t(x_i) = +1} w_{i}^{(t)} \cdot e^{- \rho_t} 
+ \sum_{y_i h_t(x_i) = -1} w_{i}^{(t)} \cdot e^{\rho_t}
\\
&= 
\sum_{i} w_{i}^{(t)} \cdot e^{- \rho_t}
+ \sum_{y_i h_t(x_i) = -1} w_{i}^{(t)} \cdot (e^{\rho_t} - e^{-\rho_t})

\end{align}
$$



(식 5)에서 `h_t`에 의존하는 것은 두 번째 항 뿐이므로, 이전 단계에서 예측에 실패한 데이터들의 Error를 최소화하는 모델을 h_t로 선택해야 한다 (즉, 이 방향으로 학습하게 해야한다). **그래서 이 방법을 Adaptive Boosting이라고 부른다.**

<br/>

##### 2-1-B. weak model의 가중치 `ρ`는 어떤 방식으로 선택하나?


$$
\frac{dE_t(X)}{d\rho_t}
=
\sum_{y_i h_t(x_i) = +1} w_{i}^{(t)} \cdot e^{- \rho_t} 
+ \sum_{y_i h_t(x_i) = -1} w_{i}^{(t)} \cdot e^{\rho_t}
=0
$$

$$
\rho_t = \frac{1}{2} \ln (
	\frac{
		\sum_{y_i h_t(x_i) = +1} w_{i}^{(t)}
	}{
		\sum_{y_i h_t(x_i) = -1} w_{i}^{(t)}
	}
)
$$


<br/>

#### 2-2. Gradient Boosting

##### 2-2-A. weak model h를 어떻게 학습하나?

Gradient Boost는 각 단계의 weak model이 이전 단계의 boosted model이 만든 residual을 예측하도록 학습한다. 즉, 다음 단계의 weak model은 이전 단계의 boosted model이 만드는 오차를 보정하도록 학습한다. 이는 다음과 같이 진행한다:

1. `H_1(X) = Y`가 되도록 모델 `H_1`를 학습한다.
2. 모델 `H_1`에서 발생한 Residual `Y - H_1(X)`를 보정하기 위해,  `h_1(X) = Y - H_1(X)`인 모델 `h_1`을 학습한다. 이 단계의 boosted model은 `H_2 = H_1(X) + h_1(X)` 이다. 
3. 모델 `H_2`에서 발생한 Residual `Y - H_2(X)`를 보정하기 위해,  `h_2(X) = Y - H_2(X)`인 모델 `h_2`을 학습한다. 이 단계의 boosted model은 `H_3(X) = H_2(X) + h_2(X)` 이다. 
4. ...이를 반복하면 최종 모델 `F_n(X)`은 `Y`에 한없이 가까워진다.

이를 수식으로 표현하면 아래와 같다:


$$
H_t(x) = H_{t-1}(x) 
+ 
\left(
	\text{arg}\,\min\limits_{h_t}\,
	\left[
		\sum_{i=1}^{n} L(y_i, H_{t-1}(x_i) + h_t(x_i))
	\right]
\right)
(x)
$$


그런데 이를 만족하는 `h_t`를 찾는 것은 computationally infeasible하다. (딥러닝에서도 비슷한 말이 나오는데, solution 자체는 explicit하지만 솔루션을 계산하는 행위 자체가 컴퓨터로 사실상 계산이 불가능하다는 뜻이다.) 대신 steepest descent step을 취해서 솔루션을 점진적으로 찾아간다. 즉, 아래와 같이 `h_t`를 선택한다:


$$
h_t(x) = -\gamma_t 
\cdot 
\sum_{i}
\nabla_{\text{params of } H_{t-1}}L(y_i, H_{t-1}(x_i))
$$


(여기서 gamma는 그냥 step length, 즉 learning rate다.)

**(식 9)**에 표현된 바와 같이, 각 단계의 weak model을 찾는 과정은 Gradient Descent와 동일하다. **그래서 이 방법을 Gradient Boosting이라고 부른다.**

<br/>

##### 2-2-B. weak model의 가중치 `ρ`는 어떤 방식으로 선택하나?

이미 예상했겠지만, gradient boosting에서 가중치 `ρ`는 그냥 1이다.

<br/>

## Practices

---

위와 같은 내용을 구현한 라이브러리가 XGBoost, LightGBM 등이다. 이들은 위 내용을 얼마나 효율적으로 구현했는지에 따라서 갈리고, 이에 따라 정확도와 속도에 차이가 있다.

원래 공부하려고 했던 Gradient Boost의 개념은 여기까지만 보면 될 것 같다. 





