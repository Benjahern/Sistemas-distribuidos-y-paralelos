FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        python3 \
        python3-numpy \
        python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN make && make test && ./build/fft2d demo --rows 8 --cols 4 --threads 2 --output /tmp/fft-demo

CMD ["./build/fft2d", "demo"]
