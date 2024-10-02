FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# main.py 및 model.json 파일을 복사
COPY ./main.py /code/
COPY ./model.json /code/

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
