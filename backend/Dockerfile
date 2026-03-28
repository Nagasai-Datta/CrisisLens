FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

ENV HF_HOME=/data/.huggingface
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]