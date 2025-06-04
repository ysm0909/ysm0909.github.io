---
layout: post
title:  "A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes"
date:   2025-06-04 15:28:36 +0900
categories: HuggingFace
---

reference: [Hugging Face Link][hugging-face-link]

# Introduction

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image.png)

Much larger models, like PaLM would require even more resources.

e.g. BLOOM-176B → 8x 80GB A100 GPUs

So, we need to find ways to reduce these requirements while preserving the model's performance.

more information..

https://arxiv.org/abs/2208.07339

# **Common data types used in Machine Learning**

- Factors that determined model size

      : the number of its parameters, precision

| 데이터 타입 | 크기 | 정밀도 | 동적 범위 | 특징 |
| --- | --- | --- | --- | --- |
| **FP32** | 4바이트 | 높음 | 넓음 | 표준 부동소수점, 대부분 하드웨어 지원 |
| **FP16** | 2바이트 | 중간 | 좁음 | 속도 빠름, overflow/underflow 문제 |
| **BF16** | 2바이트 | 낮음 | FP32와 동일 | 큰 수 표현 가능, 정밀도 낮음 |
| **TF32** | 2.375바이트 | FP16 정밀도 + BF16 범위 | 제한적 사용 | NVIDIA Ampere에서만 사용 |
| **INT8** | 1바이트 | 낮음 | 매우 좁음 | 양자화에 사용, 추론 효율 높음 |

### 💾 모델 저장 용량 계산

- 공식: `파라미터 수 × 데이터 타입 크기(바이트)`
- 예시: BLOOM-176B 모델을 BF16(2바이트)로 저장 시 →
    
    `176 × 10⁹ × 2 bytes = 352GB`
    
- 이처럼 큰 모델은 여러 GPU에 나눠 저장해야 하므로, **양자화(quantization)** 기법이 등장.

# **Introduction to model quantization**

### 🔹 FP32 대신 FP16/BF16 사용

실험 결과, 4바이트의 FP32 대신 2바이트의 FP16 또는 BF16을 사용하면 **거의 동일한 추론 결과**를 얻을 수 있음

→ 모델 크기를 절반으로 줄일 수 있었지만, 더 낮은 정밀도로 내려가면 **추론 품질 급격히 저하**

---

### 🔹 8비트 양자화 (8-bit Quantization)

이 문제를 해결하기 위해 **8비트 양자화** 도입

→ 이는 **1바이트(8비트)** 정밀도로 모델을 표현하며, **모델 크기를 FP32 기준으로 1/4로 줄일 수 있음**

하지만 단순히 비트를 줄이는 것이 아니라, **데이터를 정수로 근사(rounding)해** 표현

---

### 🔹 양자화의 개념

양자화: **하나의 데이터 타입에서 다른 타입으로 값을 반올림**하는 것

e.g. 값이 0..9 범위를 0..4 범위로 줄이면 값 4는 2로 매핑되고, 3도 2로 매핑

이처럼 서로 다른 값이 동일한 값으로 매핑되는 **정보 손실(lossy compression)** 발생

---

### 🔹 대표적인 8비트 양자화 기법

### 1. Zero-point Quantization (제로포인트 양자화)

- 예: 실수 범위 -1.0 ~ 1.0을 정수 범위 -127 ~ 127로 매핑.
- 0.3을 예로 들면:
    
    `0.3 * 127 = 38.1` → 반올림하여 38 → 다시 복원하면 `38 / 127 = 0.2992`
    
    → **0.008의 오차** 발생 (양자화 오차)
    

### 2. Absmax Quantization (절대값 최대 양자화)

- 벡터에서 절대값 기준 최대값 추출 (예: [1.2, -0.5, ..., 5.4] → max = 5.4)
- 정수 범위 [-127, 127]와 맞춰 **스케일링 계수(scale factor)**를 계산:
    
    `127 / 5.4 = 23.5`
    
- 벡터를 해당 계수로 곱해 정수 벡터로 변환:
    - 예: [1.2, -0.5, ..., 5.4] → [28, -12, ..., 127] (e.g. 1.2*23.5=28.2 → 28)
- 복원 시에는 해당 정수를 다시 스케일링 계수로 나눠 원래 값에 근접하게 계산 (정밀도 손실 존재)

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image%201.png)

---

### 🔹 기타 양자화 기술

- **unsigned int8**의 경우는 **최솟값을 뺀 후, 최대값 기준으로 스케일링**함.
- **min-max scaling**과 유사하지만, "0"을 정수 0으로 정확히 매핑하는 것이 특징.

---

### 🔹 행 단위, 벡터 단위 양자화 (Vector-wise Quantization)

- 행렬곱 A × B = C 에 대해:
    - A의 각 행, B의 각 열마다 **개별적인 최대값**을 찾아 스케일링
    - 나중에 C를 FP16으로 복원할 때는 A와 B의 최대값 벡터로 **외적(outer product)**을 계산해 복원
- 이 방식은 **LLM.int8() 논문**에 기반하여 성능 저하 없이 정밀도와 압축을 모두 달성

---

### 🔹 LLM.int8()의 중요성

기존 8비트 양자화는 **대규모 모델에서 정확도 하락** 문제가 있었지만,

**LLM.int8()**은 BLOOM-176B와 같은 **초거대 모델에서도 성능 저하 없이 작동**하는 **최초의 기법**입니다.

이는 Hugging Face의 Transformers 및 Accelerate 라이브러리에 통합되어 있습니다.

| 개념 | 설명 |
| --- | --- |
| **FP32 vs FP16/BF16** | 반 정밀도(2바이트)로 모델 추론 시 거의 동일한 성능을 유지하면서 메모리 절반 절약 가능 |
| **8-bit Quantization** | 모델 크기를 1/4로 줄이는 압축 방식, 정보 손실은 있지만 연산 속도 및 메모리 효율 증가 |
| **Zero-point Quantization** | 정규화 후 정수로 변환하는 방식, 정밀도 손실 발생 |
| **Absmax Quantization** | 텐서 내 절대값 최대를 기준으로 스케일링, 더 정밀한 양자화 가능 |
| **Vector-wise Quantization** | 각 벡터별로 따로 양자화하여 더 정확하게 행렬곱을 수행 |
| **LLM.int8()** | 성능 저하 없이 초대형 모델을 8비트로 양자화하는 혁신적 기법 |

## **A gentle summary of LLM.int8(): zero degradation matrix multiplication for Large Language Models**

LLM.int8()에서는 대형 트랜스포머 모델에서 기존 양자화가 실패하는 이유를 이해하려면 **스케일에 따라 나타나는(스케일 의존적인) 특성**을 파악하는 것이 매우 중요

성능 저하는 **아웃라이어(특정 임계값을 넘는 큰 값들)** 때문

LLM.int8() 알고리즘은 기본적으로 다음 세 단계로 행렬곱 수행:

1. 입력된 히든 상태(hidden states)에서 **열 단위로 아웃라이어 값(특정 임계값 이상인 값들)을 추출**
2. 아웃라이어 값은 FP16으로, 아웃라이어가 아닌 값들은 int8로 각각 나누어 **행렬곱을 수행**
3. int8 결과를 다시 FP16으로 **디퀀타이즈(복원)** 하고, 아웃라이어 결과와 합산해 최종 FP16 결과 얻음

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image%202.png)

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image%203.png)

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image%204.png)

![image.png](/assets/image/2025-06-04-8-bit-Matrix-Multiplication-image/image%205.png)

### **The importance of outlier features**

**트랜스포머 기반 모델이 60억(6B) 파라미터 이상으로 커질 때, 기존 양자화 방식은 실패**

작은 모델에도 큰 아웃라이어가 존재하지만, 큰 모델에서는 트랜스포머의 모든 층(layer)에서 **체계적이고 반복적인 패턴으로 나타나는 특정 임계값 이상의 아웃라이어가 존재**

8비트 정밀도는 매우 제한적이기 때문에, 큰 값이 섞인 벡터를 양자화하면 **심각한 오류가 발생 가능**

게다가 트랜스포머 구조 특성상 모든 요소가 서로 연결되어 있어, 이러한 오류가 층을 거치며 **점점 누적되어 성능 저하 악화**

따라서, 이러한 극단적 아웃라이어를 다루기 위해 **혼합 정밀도 분해(mixed-precision decomposition)** 기법이 개발

### **Inside the MatMul**

히든 상태(hidden states)를 계산한 후, **사용자 지정 임계값(threshold)을 이용해 아웃라이어를 추출**하고, 행렬을 두 부분 나눔

- 아웃라이어 크기 절댓값이 6 이상인 값을 모두 추출하면, **전체 추론 성능이 완전히 복구됨**
- 아웃라이어 부분은 **FP16(반 정밀도)**로 행렬곱을 수행하여 고전적 방식으로 처리
- 나머지 부분은 **8비트 양자화**(벡터 단위 양자화: 히든 상태는 행 단위, 가중치는 열 단위)로 처리
- 마지막에 8비트 연산 결과를 다시 FP16으로 디퀀타이즈(복원)하여 아웃라이어 연산 결과와 합산

[hugging-face-link]: https://huggingface.co/blog/hf-bitsandbytes-integration
