FROM python:3.11-slim

RUN useradd -m -u 1000 user

USER user

ENV PATH=/home/user/.local/bin:$PATH \
    HOME=/home/user \
    HF_HOME=/data/.huggingface

WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]