FROM pytorch/pytorch

WORKDIR /home/app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 9000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000", "--reload"]

