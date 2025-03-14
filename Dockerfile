
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ARG DEBIAN_FRONTEND=noninteractive  
ARG TARGETARCH  
  
# 設置工作目錄  
WORKDIR /app  
  
# 复制 app 資料夾到 Docker 映像中的 /app 目錄  
COPY . /app  

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 vim ffmpeg zip unzip htop screen tree build-essential gcc g++ make unixodbc-dev curl python3-dev python3-distutils wget libvulkan1 libfreeimage-dev \  
    && apt-get clean && rm -rf /var/lib/apt/lists  

# 安裝 python packages  
RUN pip3 install -r requirements.txt  
  
# 安裝 argos 的語言包  
RUN argospm update  
RUN argospm install translate  
  
# 設置環境變量  
ENV LC_ALL=C.UTF-8  
ENV LANG=C.UTF-8  
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility  
ENV NVIDIA_VISIBLE_DEVICES=all  
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:/usr/lib/llvm-10/lib:$LD_LIBRARY_PATH  
