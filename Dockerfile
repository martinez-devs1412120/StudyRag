FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.fileWatcherType none"]
