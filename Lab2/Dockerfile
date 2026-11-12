# Dockerfile para Laboratorio 2: Programación GPGPU con CUDA (Filtrado 2D y Sobel)
# Usa la imagen oficial con CUDA Toolkit (nvcc) para garantizar el smoke build CUDA completo
# y ejecuta la suite de pruebas (make test) + generación de paneles y gráficos.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        make \
        python3 \
        python3-numpy \
        python3-matplotlib && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Compila todo el código C++/CUDA con nvcc, ejecuta los tests automáticos
# y corre una demostración verificando la generación de figuras PNG.
RUN make clean && \
    make all && \
    make test && \
    ./build/image_cuda demo --size 64 --output /tmp/lab2-demo && \
    python3 scripts/plot_results.py --input /tmp/lab2-demo

CMD ["./build/image_cuda", "demo"]

