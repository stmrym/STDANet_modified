FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
  ENV DEBIAN_FRONTEND=noninteractive


  RUN apt-get update && apt-get upgrade -y
  RUN apt-get install -y libgl1-mesa-dev libglib2.0-0
  RUN apt-get install -y python3 python3-pip
  RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
  
  RUN pip3 install -r requirements.txt
  RUN pip3 install -U openmim
  RUN mim install mmcv-full
  RUN pip3 install mmflow
  RUN apt-get install python3-tk
  RUN cd STDAN_modified/models/ops
  RUN python3 setup.py build install 


  WORKDIR /work

  