#include "Convolution2D.h"
#include "Sobel.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(float actual, float expected, const char* message) {
    check(std::isfinite(actual) && std::fabs(actual - expected) < 1e-5f, message);
}
void impulseAndRectangularImage() {
    const float kernel[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    ImageBuffer impulse(5, 5);
    impulse.fillCenterImpulse();
    Convolution2D conv(impulse);
    conv.convolveCpu(kernel, 3);
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            const float expected = (i >= 1 && i <= 3 && j >= 1 && j <= 3)
                ? kernel[(i - 1) * 3 + j - 1] : 0.0f;
            near(conv.outputCpu()(i, j), expected, "impulso: kernel reflejado o desplazado");
        }
    }
    // Resultado calculado a mano, independiente de los bucles del algoritmo.
    ImageBuffer rectangle(2, 3);
    for (std::size_t p = 0; p < 6; ++p) rectangle.data()[p] = static_cast<float>(p + 1);
    const float expected[] = {26, 56, 54, 62, 119, 102};
    conv.setInput(rectangle);
    conv.convolveCpu(kernel, 3);
    for (std::size_t p = 0; p < 6; ++p) {
        near(conv.outputCpu().data()[p], expected[p], "convolucion rectangular/perimetro incorrecto");
        near(conv.input().data()[p], rectangle.data()[p], "se modifico la entrada");
    }
}
void blurAndSmallImages() {
    for (int side : {3, 5, 7}) {
        ImageBuffer ones(9, 11, 1.0f);
        Convolution2D conv(ones);
        const auto box = ImageBuffer::makeBoxKernel(side);
        conv.convolveCpu(box.data(), side);
        const float count = static_cast<float>((side / 2 + 1) * (side / 2 + 1));
        for (const auto corner : {std::size_t(0), std::size_t(10), std::size_t(88), std::size_t(98)})
            near(conv.outputCpu().data()[corner], count / (side * side), "box: esquina incorrecta");
        near(conv.outputCpu()(4, 5), 1.0f, "box no preserva constante interior");
        const auto gaussian = ImageBuffer::makeGaussianKernel(side);
        conv.convolveCpu(gaussian.data(), side);
        near(conv.outputCpu()(4, 5), 1.0f, "gaussiano no normalizado");
        conv.setInput(ImageBuffer(1, 1, 2.0f));
        conv.convolveCpu(box.data(), side);
        near(conv.outputCpu()(0, 0), 2.0f / (side * side), "kernel mayor que imagen");
    }
    for (std::size_t n : {8u, 16u, 32u}) {
        const auto image = ImageBuffer::createRandom(n, n, 42);
        Convolution2D conv(image);
        // Kernel direccional: salida(i,j) = entrada(i-1,j+1), incluyendo bordes.
        const float shift[] = {0, 0, 0, 0, 0, 0, 1, 0, 0};
        conv.convolveCpu(shift, 3);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                near(conv.outputCpu()(i, j), i > 0 && j + 1 < n ? image(i - 1, j + 1) : 0.0f,
                     "kernel direccional: signo, indexacion o borde incorrecto");
        const float identity = 1.0f;
        conv.convolveCpu(&identity, 1);
        for (std::size_t p = 0; p < image.size(); ++p)
            near(conv.outputCpu().data()[p], image.data()[p], "s=1 o repeticion incorrectos");
    }
}
void sobelSignsAndMagnitude() {
    for (bool vertical : {true, false}) {
        ImageBuffer image(8, 8);
        if (vertical) image.fillVerticalStep(); else image.fillHorizontalStep();
        Sobel sobel(image);
        sobel.computeGxGyCpu();
        sobel.computeMagnitudeCpu();
        near(sobel.gxCpu()(3, 3), vertical ? -4.0f : 0.0f, "signo Gx incorrecto");
        near(sobel.gyCpu()(3, 3), vertical ? 0.0f : -4.0f, "signo Gy incorrecto");
        near(sobel.magnitudeCpu()(3, 3), 4.0f, "magnitud del escalon incorrecta");
        for (std::size_t p = 0; p < image.size(); ++p) {
            const float gx = sobel.gxCpu().data()[p], gy = sobel.gyCpu().data()[p];
            near(sobel.magnitudeCpu().data()[p], std::sqrt(gx * gx + gy * gy), "magnitud incorrecta");
        }
        // Reutilizar Sobel con otra forma no debe conservar gradientes anteriores.
        sobel.setInput(ImageBuffer(3, 5));
        sobel.computeGxGyCpu();
        sobel.computeMagnitudeCpu();
        check(sobel.magnitudeCpu().height() == 3 && sobel.magnitudeCpu().width() == 5,
              "setInput no actualiza dimensiones Sobel");
        for (std::size_t p = 0; p < 15; ++p)
            near(sobel.magnitudeCpu().data()[p], 0.0f, "Sobel conserva datos anteriores");
    }
}
void invalidAndEmptyInputs() {
    Convolution2D conv(ImageBuffer(2, 2, 1));
    const float identity = 1;
    conv.convolveCpu(&identity, 1);
    for (int side : {0, -1, 2, 4}) {
        bool threw = false;
        try { conv.convolveCpu(&identity, side); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "acepta lado invalido");
    }
    bool threw = false;
    try { conv.convolveCpu(nullptr, 3); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "acepta kernel nulo");
    near(conv.outputCpu()(0, 0), 1, "entrada invalida destruye salida previa");
    conv.setInput(ImageBuffer());
    conv.convolveCpu(&identity, 1);
    check(conv.outputCpu().empty(), "entrada vacia conserva salida anterior");
    Sobel sobel;
    sobel.computeGxGyCpu();
    sobel.computeMagnitudeCpu();
    check(sobel.magnitudeCpu().empty(), "Sobel vacio incorrecto");
}
} // namespace

int main() {
    try {
        impulseAndRectangularImage();
        blurAndSmallImages();
        sobelSignsAndMagnitude();
        invalidAndEmptyInputs();
        std::cout << "[PASS] Rol 2: impulso, rectangulo, blur 3/5/7, bordes, signos Sobel, magnitud y validacion.\n";
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] Rol 2: " << error.what() << '\n';
        return 1;
    }
}
