---
layout: post
title:  "Llama Test 1(Meta)"
date: 2025-06-05 17:41:36 +0900
categories: Llama
---

reference: [Meta llama3.1 직접 다운로드 하기][link]

1. **Meta 사이트 접속**
    
    [Download Llama][link2]
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image.png)
    
    https://llama3-1.llamameta.net/*?Policy= ~ 생략
    
2. **메일 확인 → 사이트 접속**
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%201.png)
    
3. **터미널 접속**
    
    **Download**
    
    ```python
    pip install llama-stack
    ```
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%202.png)
    
    ```python
    llama model list
    ```
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%203.png)
    
    모델 선택: Llama3.1-8B
    
    ```python
    lama download --source meta --model-id Llama3.1-8B
    ```
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%204.png)
    
    받은 url 입력
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%205.png)
    
    **Delete**
    
    ```python
    find ~ -type d -name "Llama3"
    ```
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%206.png)
    
    ```python
    rm -rf /home/seoin/.llama/checkpoints/Llama3.1-8B
    ```
    
    **Way2 (llama3 설치)**
    
    [라마(LLama3.*) - Meta 웹사이트에서 직접 다운로드 및 리눅스 서버에서 운영][link3]
    
    1. **터미널 접속**
    
    실습 디렉토리: cd /home/seoin/Documents/llama_test
    
    **Download**
    
    ```python
    sudo apt-get install wget
    sudo apt-get install md5sum
    ```
    
    ```python
    git clone https://github.com/meta-llama/llama3.git
    
    cd llama3
    
    ls
    ```
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%207.png)
    
    [setup.py](http://setup.py) 활용해 필요한 패키지 종속성 설치
    
    ```python
    pip install -e .
    ```
    
    ```python
    ./download.sh
    ```
    
    Enter the URL from email: 받은 url 입력
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%208.png)
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%209.png)
    
    필요한 모델 다운로드: 8B,8B-instruct
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%2010.png)
    
    다운로드 확인
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%2011.png)
    
    **Test**
    
    1. **example_text_completion.py**
    
    ```python
    def main(
        ckpt_dir: str,              # 모델 체크포인트 디렉토리 경로
        tokenizer_path: str,        # 토크나이저 파일 또는 디렉토리 경로
        temperature: float = 0.6,   # 생성 다양성 조절 (낮을수록 더 결정적인 출력) 낮을수록 덜 무작위
        top_p: float = 0.9,         # 상위 p 누적 확률의 토큰에서 샘플링 (nucleus sampling)
        max_seq_len: int = 128,     # 입력 시퀀스의 최대 길이
        max_gen_len: int = 64,      # 생성할 최대 토큰 수
        max_batch_size: int = 4,    # 한 번에 처리할 최대 입력 개수
    ):
    ```
    
    example_text_completion.py 실행
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%2012.png)
    
    ```python
    torchrun --nproc_per_node 1 example_text_completion.py --ckpt_dir Meta-Llama-3-8B/ --tokenizer_path Meta-Llama-3-8B/tokenizer.model --max_seq_len 128 --max_batch_size 4 
    ```
    
    - `torchrun`은 분산 학습이나 멀티 GPU 환경에서 PyTorch 스크립트를 실행할 때 사용
    - `-nproc-per-node=1`이면 **GPU 한 개만 사용**
    
    실행 결과
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%2013.png)
    
    밑줄: 프롬프트, > : 답변
    
    1. **example_chat_completion.py**
    
    ```python
    def main(
        ckpt_dir: str,              # 모델 체크포인트가 저장된 디렉토리 경로
        tokenizer_path: str,        # 토크나이저 파일 또는 디렉토리 경로
        temperature: float = 0.6,   # 생성 텍스트의 다양성을 조절 (낮을수록 보수적)
        top_p: float = 0.9,         # 누적 확률 p 이하의 토큰들 중에서 샘플링 (nucleus sampling)
        max_seq_len: int = 512,     # 입력 시퀀스의 최대 길이
        max_batch_size: int = 4,    # 한 번에 처리할 최대 배치 크기
        max_gen_len: Optional[int] = None,  # 생성할 최대 토큰 수 (None이면 자동 결정)
    ):
    ```
    
    대화 예제
    
    ```python
    dialogs: List[Dialog] = [
            [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
            [
                {"role": "user", "content": "I am going to Paris, what should I see?"},
                {
                    "role": "assistant",
                    "content": """\
    ```
    
    답변 생성해서 출력하는 부분
    
    ```python
    results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
    ```
    
    example_chat_completion.py 실행
    
    ![image.png](/assets/image/2025-06-05-LlamaTest-Meta-Image/image%2014.png)
    
    ```python
    torchrun --nproc_per_node 1 example_chat_completion.py --ckpt_dir Meta-Llama-3-8B/ --tokenizer_path Meta-Llama-3-8B/tokenizer.model --max_seq_len 512 --max_batch_size 6 
    ```


[link]: https://pagichacha.tistory.com/328
[link2]: https://www.llama.com/llama-downloads/
[link3]: https://www.youtube.com/watch?v=lJwkv3J6o54
