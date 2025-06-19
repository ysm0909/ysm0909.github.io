---
layout: post
title:  "Llama Fine-tuning Test(Unsloth)"
date: 2025-06-19 13:13:36 +0900
categories: Llama
---

reference: [Unsloth][link]

# **⚙️** Setting

## ✅ 1. 가상환경 생성**❗**

```bash
#step1
conda create -n unsloth_env python=3.11 -y

#step2
conda activate unsloth_env

#step3
# PyTorch + CUDA 12.1 (pip로 설치, 공식 wheel)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ✅ 2. CUDA 설치 여부 확인

```bash
nvcc --version
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image.png)

## ✅ 3. PyTorch가 GPU를 사용하는지 확인

```python
import torch
print("CUDA 사용 가능 여부: ", torch.cuda.is_available())
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "사용불가")
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%201.png)

## ✅ 4. PyTorch GPU 버전 설치 확인

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%202.png)

⚠️ 호환 버전 확인 ⬇️

For other torch versions, we support `torch211`, `torch212`, `torch220`, `torch230`, `torch240` and for CUDA versions, we support `cu118` and `cu121` and `cu124`. 

## ⚠️ VS Code 경고 해결

```
We noticed you're using a conda environment. If you are experiencing issues with this environment in the integrated terminal, we recommend that you let the Python extension change "terminal.integrated.inheritEnv" to false in your user settings
```

➡️ VS Code에서 열려 있는 터미널이 **conda 가상환경을 제대로 인식하지 못할 수도 있다**는 의미

- `Ctrl + Shift + P` → `Preferences: Open User Settings (JSON)`

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%203.png)

# 1️⃣ Installation

```bash
pip install unsloth
```

❕python 환경: 3.10 이상

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%204.png)

➡️ torchaudio 사용할 일 없을 거 같으니 일단 보류

# 2️⃣ **Unsloth**

## ➡️ `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`  사용!

from datasets import load_dataset
dataset = load_dataset("hpe-ai/medical-cases-classification-tutorial", split="train")

```python
# ✅ Unsloth에서 제공하는 빠른 LLM 로딩 기능을 사용하기 위해 import
from unsloth import FastLanguageModel

# ✅ PyTorch는 텐서 계산, GPU 연산 등을 위한 핵심 라이브러리
import torch

# ✅ 최대 토큰 길이 설정 (예: 하나의 문장이 2048 토큰까지 가능)
max_seq_length = 2048  # 원하는 길이로 설정 가능! 내부적으로 RoPE Scaling 지원됨

# ✅ 데이터 타입 설정
# - None으로 두면 자동 감지됨
# - float16은 T4, V100 GPU에 적합
# - bfloat16은 A100, RTX 3090, 4090 등 Ampere 이상에 적합
dtype = None

# ✅ 4bit 양자화 사용 여부
# - True로 하면 메모리 사용량이 줄고 학습 및 추론이 더 빨라짐
load_in_4bit = True

# ✅ 모델과 토크나이저 로딩
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",  # 사용할 모델 이름
    max_seq_length = max_seq_length,          # 최대 입력 길이
    dtype = dtype,                            # float16, bfloat16, None 중 선택
    load_in_4bit = load_in_4bit,              # 4bit 양자화 사용 여부(양자화 모델: 다시 양자화하는 게 아니라, 올바르게 로딩하는 용도)
    token = "hf_..."                        # 허깅페이스의 접근 토큰 (Gated 모델인 경우 필요)
)
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%205.png)

```
# ✅ 미리 4bit로 양자화된 모델 리스트
# - 다운로드 속도가 빠르고 메모리 부족(OOM) 문제도 줄어듦
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama 3.1 - 8B 모델, 2배 빠른 성능
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 405B 모델도 4bit로 지원!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # 새 버전 Mistral 12B 모델, 성능 향상
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v0.3 버전
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi 3.5 모델
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Google Gemma 모델도 지원됨
]  # 전체 모델 목록은 https://huggingface.co/unsloth 참고
```

### 📁 `FastLanguageModel.from_pretrained`

- Hugging Face에 있는 모델을 **빠르고 효율적으로 불러오기 위한 함수**
- Unsloth가 내부적으로 4bit 지원, 토크나이저 최적화, GPU 최적화 등 처리

```python
# LoRA adapters 
model = FastLanguageModel.get_peft_model(
    model,
    
    # LoRA의 rank 값. 클수록 표현력은 올라가지만 VRAM 사용량도 증가합니다.
    r = 16,  # 8, 16, 32, 64, 128 등 사용 가능

    # LoRA를 적용할 대상 모듈들. Transformer의 핵심 연산 부분입니다.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],

    # LoRA의 scaling factor로, 보통 r과 같은 값을 주는 것이 일반적입니다.
    lora_alpha = 16,

    # LoRA에 적용할 dropout 비율. 0으로 설정하면 성능 및 속도 최적화에 유리합니다.
    lora_dropout = 0,  # 어떤 값이든 가능하나, 0이 가장 최적화되어 있음

    # bias 학습 여부. "none"이면 bias 파라미터는 학습하지 않음 (성능/속도 최적화됨)
    # bias: 뉴런이 더 다양한 값을 출력할 수 있게 해주는 추가적인 상수
		# LoRA에서는 대부분 학습 대상에서 제외시켜도 무방
    bias = "none",

    # ✅ "unsloth"의 커스텀 체크포인팅 방식은 VRAM을 30% 절약하고, batch size를 2배 키울 수 있음
    use_gradient_checkpointing = "unsloth",  # True 또는 "unsloth" 사용 가능

    # 랜덤성 고정을 위한 시드값 설정 (재현성 보장)
    random_state = 3407,

    # LoRA의 변형 기법 중 하나인 Rank-Stabilized LoRA 사용 여부
    use_rslora = False,  # 일반적인 LoRA 사용

    # LoftQ라는 또 다른 양자화 학습 기법 설정 (사용하지 않으면 None으로 설정)
    loftq_config = None,
)
```

# 3️⃣ Data Prep

```python
# Alpaca 형식의 프롬프트 템플릿 정의
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# 모델의 EOS (End of Sequence) 토큰을 가져옴 – 텍스트 끝 표시용
EOS_TOKEN = tokenizer.eos_token  # 반드시 EOS_TOKEN을 추가해야 함 (텍스트가 무한히 생성되지 않도록 함)

# 데이터셋의 각 예제를 Alpaca 프롬프트 형식으로 변환하는 함수 정의
def formatting_prompts_func(examples):
    # 분류 모델 생성위해 Instruction 정의
    instructions = "Classify the following medical transcription into the correct medical specialty."
    inputs       = examples["input"]        # input 열 가져오기
    outputs      = examples["output"]       # output 열 가져오기
    
    # instruction, input, output을 프롬프트 형식에 맞게 채우고 EOS 토큰 추가
    texts = [
        alpaca_prompt.format(instruction, input_, output_) + EOS_TOKEN
        for input_, output_ in zip(inputs, outputs)
    ]
    return {"text": texts} # HuggingFace datasets 포맷에 맞게 딕셔너리 형태로 반환
# 데이터셋 불러오기 (train split만 사용)
from datasets import load_dataset
dataset = load_dataset("hpe-ai/medical-cases-classification-tutorial", split="train")

# map 함수를 이용해 데이터셋 전체에 formatting_prompts_func 적용
# batched=True 설정 시 한 번에 여러 샘플을 처리 가능 (성능 향상)
dataset = dataset.map(formatting_prompts_func, batched = True)

```

# 4️⃣ **Train the model**

```python
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,                          # 파인튜닝할 사전학습된 언어 모델
    tokenizer = tokenizer,                  # 모델에 맞는 토크나이저
    train_dataset = dataset,                # 학습에 사용할 데이터셋
    dataset_text_field = "text",            # 데이터셋 내 텍스트가 있는 필드 이름
    max_seq_length = max_seq_length,       # 최대 시퀀스 길이 (토큰 수)
    dataset_num_proc = 2,                   # 데이터 전처리에 사용할 프로세스 개수 (병렬 처리)
    packing = False,                        # 짧은 시퀀스 붙여서 배치 처리하는 기능 (속도 향상용), 여기선 비활성화
    args = TrainingArguments(
        per_device_train_batch_size = 8,   # 한 GPU 당 배치 크기
        gradient_accumulation_steps = 4,   # 그래디언트 누적 스텝 수 (실질 배치 크기 = 2 * 4 = 8)
        warmup_steps = 5,                  # 학습 초기 학습률 점진적 증가 기간
        # num_train_epochs = 1,            # 전체 에포크 수 (주석 처리됨, 대신 max_steps 사용)
        max_steps = 60,                    # 최대 학습 스텝 수 (60 스텝까지만 학습)
        learning_rate = 2e-4,             # 학습률
        fp16 = not is_bfloat16_supported(),   # bf16 미지원 시 fp16 사용 (혼합 정밀도)
        bf16 = is_bfloat16_supported(),        # bf16 지원 시 활성화
        logging_steps = 1,                # 몇 스텝마다 로그 기록할지 (여기선 매 스텝)
        optim = "adamw_8bit",             # 8비트 양자화된 AdamW 옵티마이저 사용 (메모리 절약 목적)
        weight_decay = 0.01,              # 가중치 감쇠 계수 (정규화)
        lr_scheduler_type = "linear",     # 학습률 스케줄러 (선형 감소)
        seed = 3407,                      # 재현성을 위한 시드 설정
        output_dir = "outputs",           # 체크포인트 저장 폴더 경로
        report_to = "none",               # WandB 등 외부 로그 비활성화
    ),
)

```

```python
trainer_stats = trainer.train()
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%206.png)

# 5️⃣ **Inference**

```python
# 🧾 Prompt 형식 동일하게 유지
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ✅ 예시 문장 (Inference용)
instruction = "Classify the following medical transcription into the correct medical specialty."
transcription = "PREOPERATIVE DIAGNOSES:,1. Right axillary adenopathy.,2. Thrombocytopenia.,3. Hepatosplenomegaly.,POSTOPERATIVE DIAGNOSES:,1. Right axillary adenopathy.,2. Thrombocytopenia.,3. Hepatosplenomegaly.,PROCEDURE PERFORMED: ,Right axillary lymph node biopsy.,ANESTHESIA: , Local with sedation.,COMPLICATIONS: , None.,DISPOSITION: , The patient tolerated the procedure well and was transferred to the recovery room in stable condition.,BRIEF HISTORY: ,The patient is a 37-year-old male who presented to ABCD General Hospital secondary to hiccups and was ultimately found to have a right axillary mass to be severely thrombocytopenic with a platelet count of 2000 as well as having hepatosplenomegaly. The working diagnosis is lymphoma, however, the Hematology and Oncology Departments were requesting a lymph node biopsy in order to confirm the diagnosis as well as prognosis. Thus, the patient was scheduled for a lymph node biopsy with platelets running secondary to thrombocytopenia at the time of surgery.,INTRAOPERATIVE FINDINGS: , The patient was found to have a large right axillary lymphadenopathy, one of the lymph node was sent down as a fresh specimen.,PROCEDURE: ,After informed written consent, risks and benefits of this procedure were explained to the patient. The patient was brought to the operating suite, prepped and draped in a normal sterile fashion. Multiple lymph nodes were palpated in the right axilla, however, the most inferior node was to be removed. First, the skin was anesthetized with 1% lidocaine solution. Next, using a #15 blade scalpel, an incision was made approximately 4 cm in length transversally in the inferior axilla. Next, using electro Bovie cautery, maintaining hemostasis, dissection was carried down to the lymph node. The lymph node was then completely excised using electro Bovie cautery as well as hemostats to maintain hemostasis and then lymph node was sent to specimen fresh to the lab. Several hemostats were used, suture ligated with #3-0 Vicryl suture and hemostasis was maintained. Next the deep dermal layers were approximated with #3-0 Vicryl suture. After the wound has been copiously irrigated, the skin was closed with running subcuticular #4-0 undyed Vicryl suture and the pathology is pending. The patient did tolerated the procedure well. Steri-Strips and sterile dressings were applied and the patient was transferred to the Recovery in stable condition."
# ✅ 프롬프트 생성 (output은 빈칸)
prompt = alpaca_prompt.format(instruction, transcription, "")

# ✅ 토크나이즈 + 추론
from transformers import AutoTokenizer

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

# ⚡ 2배 속도 옵션 (LoRA 적용 모델인 경우에만 사용 가능)
# FastLanguageModel.for_inference(model)  # PEFT 모델에서는 보통 생략 가능

outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

# ✅ 결과 디코딩
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%207.png)

# 6️⃣ **Test Evaluation**

```python
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np

# Alpaca prompt 템플릿
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

instruction = "Classify the following medical transcription into the correct medical specialty."

# 라벨 전처리 함수
def normalize_label(label):
    return label.lower().replace("-", "/").replace(" / ", "/").strip()

# 입력 길이 제한용 토큰 수
MAX_INPUT_TOKENS = 1800

# Test dataset 로드
test_dataset = load_dataset("hpe-ai/medical-cases-classification-tutorial", split="test")

labels, preds = [], []

# 라벨 집합 확보 (정규화된 기준으로)
all_classes = sorted(list({normalize_label(x["medical_specialty"]) for x in test_dataset}))

for example in tqdm(test_dataset):
    transcription = example["transcription"]
    true_label = normalize_label(example["medical_specialty"])
    labels.append(true_label)

    prompt = alpaca_prompt.format(instruction, transcription, "")

    # 토크나이징 + 길이 제한
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=32, use_cache=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # "### Response:" 이후 예측값만 추출
    response = output_text.split("### Response:")[-1].strip().split("\n")[0]
    pred = normalize_label(response)
    preds.append(pred)

# 정확도 및 리포트 출력
print("✅ Accuracy:", accuracy_score(labels, preds))
print("\n📄 Classification Report:")
print(classification_report(labels, preds, labels=all_classes))

# Confusion matrix
cm = confusion_matrix(labels, preds, labels=all_classes)

plt.figure(figsize=(18, 14))
sns.heatmap(cm, xticklabels=all_classes, yticklabels=all_classes, annot=False, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%208.png)

![image.png](/assets/image/2025-06-19-Llama-Fine-tuning-Test-Unsloth-Image/image%209.png)

# 7️⃣ **Saving, loading finetuned models**

```python
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")
# model.push_to_hub("your_name/lora_model", token = "...") # Online saving
# tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving
```


[link]: https://github.com/unslothai/unsloth
