FROM python:3.11-slim

WORKDIR /home/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--reload" , "--port","9000"]
