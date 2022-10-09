FROM python:3.7

WORKDIR /webapp/
COPY ./webapp/requirements.txt requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT [ "python3.7" ]
CMD [ "./webapp/app.py" ]
