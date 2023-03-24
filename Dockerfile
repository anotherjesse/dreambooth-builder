from r8.im/andreasjansson/stable-diffusion-inpainting

RUN pip install diffusers==0.6.0
RUN pip install torch==1.13.1 --extra-index-url=https://download.pytorch.org/whl/cu117
RUN pip install ftfy==6.1.1
RUN pip install scipy==1.9.0
RUN pip install transformers==4.21.1


copy predict.py /src/predict.py
copy weights /src/weights