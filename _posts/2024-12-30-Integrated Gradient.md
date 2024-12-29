---
title: Integrated Gradient (Axiomatic Attribution for Deep Networks)
date: 2024-12-30 03:00:00 +0900
categories: [Machine Learning]
tags: [Machine Learning, XAI]
---

**Paper**: Axiomatic Attribution for Deep Networks (https://arxiv.org/abs/1703.01365)

내용 요약보다는 내가 이 논문을 어떤 방향으로 이해했는지에 대해 적었다.

<br/>

## 핵심 아이디어

기본적으로 이 논문은 신경망의 결과물을 Scalar Function으로 보고, 신경망의 Gradient를 Conservative Vector Field로 생각한다. 그리고 이를 통해 신경망의 예측값을 수식적으로 어떻게 표현할 수 있는지 파악하여 XAI로 활용한 것에 불과하다. 

즉, 논문 전체의 아이디어는 하나의 수식이고, 나머지는 이 수식을 정당화하기 위한 문제 제기, 그리고 검증의 과정이다. 수식 자체도 굉장히 단순하며, (논문에서 말하는 바와 같이) 완전한 초보자도 사용할 수 있는 수식이다.

<br/>

## 주의 사항

1. baseline을 잘 정해야 한다. 모델에서는 F(baseline) ~ neutral 인 baseline을 추천한다. 
   (논문에서 조금 헷갈리게 해놓은게, 신경망 예측값의 범위는 [0, 1]로 설명했지만 neutral인 값은 0으로 표시했다. Integrated Gradient의 의도를 따라가려면 F(baseline) ~ 0으로 하는게 맞다.)
2. NLP에서 baseline은 encoded index가 아니라 embedding을 의미한다. 즉, origin에 있는 embedding vector를 의미한다. 이 점에서 Integrated Gradient를 Embedding Layer가 있는 신경망에 적용할 때 embedding layer에만 먼저 적용한 값들을 사용해야 하는 불편함이 있다. (이 점에서 완전한 초보자만 사용할 수 있다는 건 조금 과장같다.)

<br/>

<img src="./../assets/img/figs/2024-12-30-Integrated Gradient/fig01.jpg" alt="fig01" style="zoom: 50%;" />
