FROM apache/airflow:2.8.3-python3.8
COPY requirements.txt /requirements.txt
COPY dags/data ./opt/airflow/dags/data
RUN mkdir -p /home/airflow/.kaggle
COPY dags/kaggle.json /home/airflow/.kaggle/kaggle.json
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Cambia el usuario de nuevo a airflow
USER airflow
RUN pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt

