---
layout: post
title:  "Llama Test 2(GGUF+Docker)"
date: 2025-06-10 11:13:36 +0900
categories: Llama
---

reference: [A Quick Guide to Containerizing Llamafile with Docker for AI Applications (Sophia Parafina)][link]

## **1. To get started, copy, paste, and save the following in a file named Dockerfile.**

Debian Trixie를 기반으로 Mozilla-Ocho의 [llamafile](https://github.com/Mozilla-Ocho/llamafile) 프로젝트를 빌드하고, 가벼운 이미지로 패키징하여 실행 가능한 컨테이너를 구성하는 것

```docker
# < 1단계: 빌드 스테이지로 Debian Trixie를 사용 (GCC 13 사용 가능) >
# Use debian trixie for gcc13
FROM debian:trixie as builder
 
# Set work directory
WORKDIR /download
 
# Configure build container and build llamafile
RUN mkdir out && \
    apt-get update && \
    apt-get install -y curl git gcc make && \ # 필요한 도구 설치
    git clone https://github.com/Mozilla-Ocho/llamafile.git  && \ # 소스 코드 클론
    curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \ # 커스텀 unzip 도구 다운로드
    chmod 755 unzip && mv unzip /usr/local/bin && \ # 실행 권한 부여 및 경로 이동
    cd llamafile && make -j8 LLAMA_DISABLE_LOGS=1 && \ # 로깅 비활성화하고 병렬 빌드 수행
    make install PREFIX=/download/out # 빌드 결과물을 지정된 경로로 설치
 
# < 2단계: 실행 스테이지로 Debian stable 사용 (더 작고 안전한 이미지) >
# Create container
FROM debian:stable as out
 
# Create a non-root user
RUN addgroup --gid 1000 user && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos "" user
 
# Switch to user
USER user
 
# Set working directory
WORKDIR /usr/local
 
# Copy llamafile and man pages
COPY --from=builder /download/out/bin ./bin
COPY --from=builder /download/out/share ./share/man
 
# Expose 8080 port.
EXPOSE 8080
 
# Set entrypoint.
ENTRYPOINT ["/bin/sh", "/usr/local/bin/llamafile"]
 
# Set default command.
CMD ["--server", "--host", "0.0.0.0", "-m", "/model"]
```

## **2. To build the container, run:**

```bash
docker build -t llamafile .
```

⛔ **Error 1**

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image.png)

**💡 Solution**

참고: [https://seulcode.tistory.com/557](https://seulcode.tistory.com/557)

1. docker group 생성
    
    ```bash
    sudo groupadd docker
    ```
    
2. docker group에 유저 추가
    
    ```bash
    sudo usermod -aG docker $USER
    ```
    
3. 아래 명령어 실행
    
    ```bash
    newgrp docker
    ```
    

⛔ **Error 2**

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%201.png)

**💡 Solution**

```docker
# Configure build container and build llamafile
RUN mkdir out && \
    apt-get update && \
    apt-get install -y curl git gcc make && \ # 필요한 도구 설치
    git clone https://github.com/Mozilla-Ocho/llamafile.git  && \ # 소스 코드 클론
    curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \ # 커스텀 unzip 도구 다운로드
    chmod 755 unzip && mv unzip /usr/local/bin && \ # 실행 권한 부여 및 경로 이동
    cd llamafile && make -j8 LLAMA_DISABLE_LOGS=1 && \ # 로깅 비활성화하고 병렬 빌드 수행
    make install PREFIX=/download/out # 빌드 결과물을 지정된 경로로 설치
```

 **여러 개의 RUN 명령어로 나눔**

```docker
# Prepare output directory
RUN mkdir /download/out
# Install required packages
RUN apt-get update && \
    apt-get install -y curl git gcc g++ make build-essential zlib1g-dev
# Clone llamafile repository
RUN git clone https://github.com/Mozilla-Ocho/llamafile.git
# Install unzip from cosmos.zip
RUN curl -L -o ./unzip https://cosmo.zip/pub/cosmos/bin/unzip && \
    chmod 755 unzip && mv unzip /usr/local/bin
# Set working directory to the repo
WORKDIR /download/llamafile
# Build llamafile
RUN make LLAMA_DISABLE_LOGS=1
# Install llamafile to output dir
RUN make install PREFIX=/download/out
```

## **3. LLamafile(GGUF) Download**

[(Huggingface) Llama-3.3-70B-Instruct-GGUF ][link2]

## **4. Running the llamafile container**

```bash
docker run -d -v ./model/<모델명.gguf>:/model -p 8080:8080 llamafile
```

➡️ model 폴더 안에 있어야 함.

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%202.png)

```
http://127.0.0.1:8080
```

➡️ llama.cpp interface 접근

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%203.png)

# 5. Test

✅ **Test 1**

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%204.png)

**✅ Test2**

```bash
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "system",
      "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
    },
    {
      "role": "user",
      "content": "Compose a poem that explains the concept of recursion in programming."
    }
  ]
}' | python3 -c '
import json
import sys
json.dump(json.load(sys.stdin), sys.stdout, indent=2)
print()
'
```

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%205.png)

**⬇️ output**

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%206.png)

✅ **Check**

- 정상 작동 여부 확인

```bash
docker ps
```

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%207.png)

- 종료된 컨테이너 확인

```bash
docker ps -a
```

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%208.png)

- 로그 확인

```bash
docker logs <컨테이너 ID>
```

➡️ 로그통해 사이트 접속 llama.cpp interface 접근 가능

![image.png](/assets/image/2025-06-10-LlamaTest-Docker-Image/image%209.png)

- 컨테이너 종료

```bash
docker stop <컨테이너 ID>
```


[link]: https://www.docker.com/blog/a-quick-guide-to-containerizing-llamafile-with-docker-for-ai-applications/
[link2]: https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/blob/main/Llama-3.3-70B-Instruct-Q3_K_M.gguf
