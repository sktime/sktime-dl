
FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN pip install tensorflow-addons==0.8.2
RUN pip install Cython==0.29.14
RUN pip install pytest==5.3.5
RUN pip install pytest-cov==2.8.1
RUN pip install flaky==3.6.1
RUN pip install sktime==0.3.1

WORKDIR /usr/src/app
COPY . sktime_dl
