FROM python:3.11-slim

WORKDIR /home/app

COPY . .

COPY packages/ /home/app/packages/

# تثبيت الباكيجات من فولدر packages فقط، بدون استخدام الإنترنت
RUN pip install --no-index --find-links=/home/app/packages/ -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]

