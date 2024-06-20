FROM python:3.9

RUN useradd -m -u 1000 user

WORKDIR /app

COPY ./startings.csv ./startings.csv
COPY ./src ./src
COPY ./requirements.txt ./requirements.txt
COPY ./main.py ./main.py

RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install pandas numpy transformers fastapi unicorn[standard]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
