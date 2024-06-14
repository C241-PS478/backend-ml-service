FROM python:3.12

# copy all files of the current directory to the folder /app
COPY . /app
WORKDIR /app

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD waitress-serve --port=${PORT:-8080} --call main:create_app