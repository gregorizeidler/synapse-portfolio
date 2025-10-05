# Dockerfile opcional para reproduzir o ambiente
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponha a porta do Streamlit
EXPOSE 8501

CMD ["bash", "-lc", "streamlit run src/dashboard.py --server.address=0.0.0.0 --server.port=8501"]
