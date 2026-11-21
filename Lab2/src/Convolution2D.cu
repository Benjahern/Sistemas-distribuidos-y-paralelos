#include "Convolution2D.h"

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <climits>
#include <stdexcept>
#include <cstdio>
#include <string>


namespace {

void checkCuda(
    cudaError_t state,
    const char* call,
    const char* file,
    int line
 ) {
    // Si el CUDA falla, mostramos qué llamada fue y donde ocurrio
    if (state != cudaSuccess) {
        throw std::runtime_error(std::string( "Error CUDA en ") + file + ":" + std::to_string(line) + " [" + call + "] " + cudaGetErrorString(state));
    }

}

#define CUDA_CHECK(call) \
    checkCuda((call), #call, __FILE__, __LINE__)
    
// Usamos este buffer para liberar la memoria GPU 
// tambien si ocurre un error y salimos antes de terminar

class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void**>(&data_),
            count * sizeof(float)
        ));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            // Aquí mostramos el error sin lanzar otra excepción
            const cudaError_t state = cudaFree(data_);

            if (state != cudaSuccess) {
                std::fprintf(stderr, 
                "Error al liberar memoria CUDA: %s\n",
            cudaGetErrorString(state));

            }
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    float* data() const noexcept {
        return data_;
    }

    void realease() {
        if (data_ == nullptr) {
            return;
        }
    

    float* previous = data_;
    data_ = nullptr;

    CUDA_CHECK(cudaFree(previous));

}

private:
    float* data_ = nullptr;

};

__device__ float readZeroPadded(
    const float* image,
    int row,
    int col,
    int height,
    int width
) {
    // Si el vecino queda fuera de la imagen, usamos cero
    if (
        row < 0 || row >= height ||
        col < 0 || col >= width
    ) {
        return 0.0f;
    }
    return image[row*width + col];
}

__global__ void convolveNaiveKernel(
    const float* input,
    float* output,
    const float* coefficients,
    int height,
    int width,
    int s
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row>= height || col >= width){
        return;
    }

    const int radius = s/2;
    float sum = 0.0f;

    int kernelRow = 0;

    while(kernelRow<s) {
        const int u = kernelRow - radius;
        int kernelCol = 0;

        while(kernelCol<s) {
            const int v = kernelCol - radius;

            // Restamos los desplazamientos para hacer la misma convolución que en CPU
            const float neighbor = readZeroPadded(
                input,
                row - u,
                col - v,
                height,
                width
            );

            sum+= coefficients[kernelRow*s + kernelCol]* neighbor;
            ++kernelCol;
        }
        ++kernelRow;
    }

    output[row * width + col] = sum;

}

__global__ void convolveSharedKernel(
    const float* input,
    float* output,
    const float* coefficients,
    int height,
    int width,
    int s
) {
    extern __shared__ float tile[];

    const int radius = s/2;

    const int blockWidth = static_cast<int>(blockDim.x);
    const int blockHeight = static_cast<int>(blockDim.y);

    const int tileWidth = blockWidth + 2 * radius;
    const int tileHeight = blockHeight + 2*radius;

    const int localX = static_cast<int>(threadIdx.x);
    const int localY = static_cast<int>(threadIdx.y);

    const int startCol = static_cast<int>(blockIdx.x)*blockWidth;
    const int startRow = static_cast<int>(blockIdx.y)*blockHeight;

    const int col = startCol + localX;
    const int row = startRow + localY;

    // Entre todos cargamos esta parte de la imagen 
    // Algunos hilos cargan más de un valor para completar el tile
    
    int tileRow = localY;

    while(tileRow < tileHeight){
        int tileCol = localX;

        while(tileCol < tileWidth){
            const int inputRow = startRow + tileRow - radius;
            const int inputCol = startCol + tileCol - radius;

            tile[tileRow*tileWidth + tileCol] = readZeroPadded(
                input,
                inputRow,
                inputCol,
                height,
                width
            );
            tileCol+=blockWidth;
        }
        tileRow+=blockHeight;
    }

    // Esperamos a que todos terminen de cargar antes de leer
    __syncthreads();

    // Comprobamos los límites después de sincronizar
    if (row >= height || col >= width){
        return;
    }

    float sum = 0.0f;

    int kernelRow = 0;

    while(kernelRow < s){
        const int u = kernelRow - radius;

        int kernelCol = 0;

        while(kernelCol<s){
            const int v = kernelCol - radius;

            const int tileRow = localY + radius - u;
            const int tileCol = localX + radius - v;

            const float neighbor = tile[tileRow*tileWidth + tileCol];

            sum+=coefficients[kernelRow*s + kernelCol]*neighbor;

            ++kernelCol;
        }
        ++kernelRow;
    }

    output[row*width+col] = sum;

}
}

void Convolution2D::convolveGpu(
    const float* k,
    int s,
    int variant,
    dim3 block
) {
    if(k==nullptr){
        throw std::invalid_argument(
            "convolveGpu: El kernel no puede ser nulo"
        );
    }

    // Validar que el lado del kernel pertenezca a los tamaños del laboratorio.
    if (s != 3 && s != 5 && s != 7){
        throw std::invalid_argument(
            "convolveGpu: s debe ser 3, 5 o 7"
        );
    }

    if (variant != VARIANT_NAIVE && variant != VARIANT_SHARED) {
        throw std::invalid_argument("convolveGpu: variante invalida");
    }

    if (block.x == 0 || block.y == 0 || block.z != 1){
        throw std::invalid_argument("convolveGpu: Se requiere un bloque 2D valido");
    }

    // Usamos índices int y dejamos margen
    // y los hilos que sobrepasen el borde de la imagen

    if (
        input_.height() > static_cast<std::size_t>(INT_MAX - 4096) ||
        input_.width() > static_cast<std::size_t>(INT_MAX - 4096) ||
        input_.size() > static_cast<std::size_t>(INT_MAX)
    ) {
        throw std::invalid_argument(
            "convolveGpu: Imagen demasiado grande para estos índices"
        );
    }

    ImageBuffer result(input_.height(), input_.width(), 0.0f);

    if (input_.empty()){
        output_gpu_ = result;
        return;
    }

    const int height = static_cast<int>(input_.height());
    const int width = static_cast<int>(input_.width());

    int device = 0;
    cudaDeviceProp properties{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties,device));

    const std::size_t threadCount = static_cast<std::size_t>(block.x) * block.y;

    // Revisamos que la GPU soporte el bloque solicitado
    if (
        block.x > static_cast<unsigned int>(
            properties.maxThreadsDim[0]
        ) || 
        block.y > static_cast<unsigned int>(
            properties.maxThreadsDim[1]
        ) ||
        threadCount > static_cast<std::size_t>(
            properties.maxThreadsPerBlock
        )
    ) {
        throw std::invalid_argument(
            "convolveGpu: Bloque no soportado por la GPU"
        );
    }

    // Redondeamos hacia arriba para cubrir toda la imagen,
    // aunque sus dimensiones no sean múltiplos del bloque
    const dim3 grid(
        (static_cast<unsigned int>(width) + block.x -1) / block.x,
        (static_cast<unsigned int>(height) + block.y -1) / block.y
    );

    if (
        grid.x > static_cast<unsigned int>(
            properties.maxGridSize[0]
        ) ||
        grid.y > static_cast<unsigned int>(
            properties.maxGridSize[1]
        )
    ) {
        throw std::invalid_argument(
            "convolveGpu: Grilla no soportada por la GPU"
        );
    }

    const int radius = s/2;

    const std::size_t sharedBytes =
        (static_cast<std::size_t>(block.x) + 2 *radius) *
        (static_cast<std::size_t>(block.y) + 2 *radius) *
        sizeof(float);

    if (
        variant == VARIANT_SHARED && sharedBytes > properties.sharedMemPerBlock
    ) {
        throw std::invalid_argument(
            "convolveGpu: Memoria compartida insuficiente"
        );
    }

    const std::size_t pixelCount = input_.size();
    const std::size_t imageBytes = pixelCount * sizeof(float);

    const std::size_t coefficientCount = static_cast<std::size_t>(s) * s;
    const std::size_t kernelBytes = coefficientCount * sizeof(float);

    DeviceBuffer deviceInput(pixelCount);
    DeviceBuffer deviceOutput(pixelCount);
    DeviceBuffer deviceKernel(coefficientCount);

    // Copiamos la imagen y los coeficientes desde CPU a GPU
    CUDA_CHECK(cudaMemcpy(
        deviceInput.data(),
        input_.data(),
        imageBytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        deviceKernel.data(),
        k,
        kernelBytes,
        cudaMemcpyHostToDevice
    ));

    if (variant == VARIANT_NAIVE){
        convolveNaiveKernel<<<grid, block>>>(
            deviceInput.data(),
            deviceOutput.data(),
            deviceKernel.data(),
            height,
            width,
            s
        );

        CUDA_CHECK(cudaGetLastError());
    } else{
        convolveSharedKernel<<<grid, block, sharedBytes>>>(
            deviceInput.data(),
            deviceOutput.data(),
            deviceKernel.data(),
            height,
            width,
            s
        );

        CUDA_CHECK(cudaGetLastError());
    }

    // Esperamos a que termine el kernel y 
    // comprobamos si ocurrio algún error durante su ejecución
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        result.data(),
        deviceOutput.data(),
        imageBytes,
        cudaMemcpyDeviceToHost
    ));

    deviceKernel.realease();
    deviceOutput.realease();
    deviceInput.realease();

    output_gpu_ = result;

}

// Benchmark/host-device entry point: launches the same kernels without owning
// buffers or performing transfers, allowing Rol 4 to keep allocations resident.
void benchmarkLaunchConvolution(const float* input, float* output,
                                const float* coefficients, std::size_t height,
                                std::size_t width, int s, int variant,
                                unsigned int blockX, unsigned int blockY) {
    if (input == nullptr || output == nullptr || coefficients == nullptr ||
        height == 0 || width == 0 || s <= 0 || s % 2 == 0 ||
        (variant != Convolution2D::VARIANT_NAIVE && variant != Convolution2D::VARIANT_SHARED) ||
        blockX == 0 || blockY == 0 || height > static_cast<std::size_t>(INT_MAX) ||
        width > static_cast<std::size_t>(INT_MAX)) {
        throw std::invalid_argument("benchmarkLaunchConvolution: argumentos inválidos");
    }
    const dim3 block(blockX, blockY);
    const dim3 grid((static_cast<unsigned int>(width) + blockX - 1) / blockX,
                    (static_cast<unsigned int>(height) + blockY - 1) / blockY);
    const std::size_t sharedBytes =
        (static_cast<std::size_t>(blockX) + 2 * (s / 2)) *
        (static_cast<std::size_t>(blockY) + 2 * (s / 2)) * sizeof(float);
    if (variant == Convolution2D::VARIANT_NAIVE) {
        convolveNaiveKernel<<<grid, block>>>(input, output, coefficients,
                                             static_cast<int>(height),
                                             static_cast<int>(width), s);
    } else {
        convolveSharedKernel<<<grid, block, sharedBytes>>>(
            input, output, coefficients, static_cast<int>(height),
            static_cast<int>(width), s);
    }
    CUDA_CHECK(cudaGetLastError());
}

#undef CUDA_CHECK

#endif
