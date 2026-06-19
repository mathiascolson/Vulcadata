FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY dashboard_config.py /app/dashboard_config.py
COPY dashboard_data.py /app/dashboard_data.py
COPY dashboard_components.py /app/dashboard_components.py
COPY .streamlit /app/.streamlit

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
