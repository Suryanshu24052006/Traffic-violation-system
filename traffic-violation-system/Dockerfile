FROM python:3.11-slim

# OpenCV / EasyOCR runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/snapshots database

EXPOSE 8501

# Default: launch the dashboard. Override CMD to run the pipeline instead:
#   docker run <image> python -m src.pipeline.main_pipeline --video ...
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
