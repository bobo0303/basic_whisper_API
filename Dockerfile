
# This docker is only for DLAP-211-ORIN-NX 16GB - Jetpack 5.1.1 [L4T 35.3.1] | ubuntu 20.04 | CUDA 11.4 | 
# If U use difference edge it may happend some error, because of cuda/cudnn version or Ctranslate2 unsupported  

FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel
  
ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils wget libvulkan1 libfreeimage-dev \  
    && apt-get clean && rm -rf /var/lib/apt/lists  

    # 複製文件  
COPY requirements.txt .  
COPY whl/openai_whisper-20240930-py3-none-any.whl .  
COPY whl/googletrans-4.0.0rc1-py3-none-any.whl .
COPY deb/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb /tmp/  
  
# 安裝 python packages  
RUN pip3 install -r requirements.txt  
  
# 安裝 argos 的語言包  
RUN argospm update  
RUN argospm install translate  
  
# 安裝 cudnn 9.1
RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb  
RUN cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-local-52C3CBCA-keyring.gpg /usr/share/keyrings/  
RUN apt-get update  
RUN apt-get -y install cudnn-cuda-12
  
# 設置工作目錄  
WORKDIR /app  
  
# 复制 app 資料夾到 Docker 映像中的 /app 目錄  
COPY . /app  
  
# 設置環境變量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-10/lib:$LD_LIBRARY_PATH  
