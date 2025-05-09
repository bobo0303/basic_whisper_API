FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel  
  
ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
WORKDIR /mnt  
  
RUN apt-get update && apt-get install -y --no-install-recommends \  
    libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils git wget libvulkan1 libfreeimage-dev \  
    && apt-get clean && rm -rf /var/lib/apt/lists/*  
  
COPY requirements.txt /tmp/requirements.txt  
COPY whl/openai_whisper-20240930-py3-none-any.whl /tmp/openai_whisper-20240930-py3-none-any.whl  
COPY whl/googletrans-4.0.0rc1-py3-none-any.whl /tmp/googletrans-4.0.0rc1-py3-none-any.whl  
COPY deb/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb /tmp/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb  
  
RUN pip3 install -r /tmp/requirements.txt  
  
# 安裝 cudnn 9.1  
RUN dpkg -i /tmp/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb \  
    && cp /var/cudnn-local-repo-ubuntu2204-9.1.0/cudnn-local-52C3CBCA-keyring.gpg /usr/share/keyrings/ \  
    && apt-get update \  
    && apt-get -y install cudnn-cuda-12  
  
# 設置環境變量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-10/lib:$LD_LIBRARY_PATH  

  
RUN rm /tmp/cudnn-local-repo-ubuntu2204-9.1.0_1.0-1_amd64.deb  
RUN rm /tmp/openai_whisper-20240930-py3-none-any.whl  
RUN rm /tmp/googletrans-4.0.0rc1-py3-none-any.whl  
RUN rm /tmp/requirements.txt  

# 進去後記得先 huggingface-cli login 
ARG HUGGINGFACE_HUB_TOKEN  
ENV HUGGINGFACE_HUB_TOKEN=${HUGGINGFACE_HUB_TOKEN}  
RUN echo "$HUGGINGFACE_HUB_TOKEN" | huggingface-cli login --token  
