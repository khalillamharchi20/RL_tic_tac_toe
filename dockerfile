FROM rayproject/ray:latest

WORKDIR /app

RUN pip install --no-cache-dir torch numpy matplotlib

