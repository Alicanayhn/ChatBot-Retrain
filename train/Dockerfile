FROM python:3.10

WORKDIR /trainer

COPY requirements.txt /trainer/requirements.txt

RUN pip install -r /trainer/requirements.txt

COPY . /trainer

CMD [ "python", "app.py" ]