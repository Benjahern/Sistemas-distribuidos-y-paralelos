#include "../include/ComplexField.h"
#include <iostream>
#include <stdexcept>

int main() {
    ComplexField f(8, 4);
    f.at(1, 2) = {3.0, 4.0};
    std::cout << f.at(1, 2) << "\n";
    std::cout << f.rows() << "x" << f.cols() << "\n";

    try {
        ComplexField malo(1, 4);
    } catch (const std::invalid_argument& e) {
        std::cout << "Error esperado: " << e.what() << "\n";
    }

    ComplexField::Complex arr[8];
    for (int i = 0; i < 8; ++i) {
        arr[i] = ComplexField::Complex(i, 0.0);
    }

    ComplexField::bitReversePermute(arr, 8);

    for (int i = 0; i < 8; i++){
        std::cout << arr[i].real() << " ";
    }
    std::cout << "\n";

    ComplexField a(8, 4);
ComplexField b(8, 4);
a.fillRandom(42);
b.fillRandom(42);
bool igual = true;
int m = 0;
while (m < 8) {
    int n = 0;
    while (n < 4) {
        if (a.at(m, n) != b.at(m, n)) igual = false;
        n++;
    }
    m++;
}
std::cout << "fillRandom reproducible: " << (igual ? "OK" : "FALLO") << "\n";

ComplexField c(4, 4);
c.fillImpulse();
std::cout << "impulso: " << ((c.at(0,0) == ComplexField::Complex(1.0, 0.0) && c.at(1,2) == ComplexField::Complex(0.0, 0.0)) ? "OK" : "FALLO") << "\n";
}