FROM python:3.9-slim

WORKDIR /

COPY . /

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
