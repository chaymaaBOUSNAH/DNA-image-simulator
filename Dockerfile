FROM python:3
# the working directory for running instances of the container image.
WORKDIR image_generator

ADD requirements.txt .

# no-cache-dir : Disable the cache.
# update pip to take advantage of the new features and security patches
RUN pip install --no-cache-dir --upgrade --pre pip
RUN pip install --no-cache-dir -r requirements.txt

ADD . .

CMD /bin/bash