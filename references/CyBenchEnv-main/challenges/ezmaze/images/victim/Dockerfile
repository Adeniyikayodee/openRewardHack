FROM python:3.9-slim-buster

RUN sed -i 's/deb.debian.org/archive.debian.org/g' /etc/apt/sources.list && \
    sed -i 's|security.debian.org/debian-security|archive.debian.org/debian-security|g' /etc/apt/sources.list && \
    sed -i '/buster-updates/d' /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y lib32z1 xinetd socat && \
    pip3 install requests pycryptodome && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m user && \
    chown -R root:root /home/user

COPY app /home/user/
RUN chmod +x /home/user/chall.py
COPY xinetd /etc/xinetd.d/user

WORKDIR /home/user

EXPOSE 1337

CMD ["/usr/sbin/xinetd", "-dontfork"]
