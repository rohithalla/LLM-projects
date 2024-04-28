FROM python:3.9
RUN apt-get update && apt-get install -y --no-install-recommends libgeos-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
EXPOSE 8080
WORKDIR /app
COPY . ./
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run", "main.py",  "--server.port=8080"] 