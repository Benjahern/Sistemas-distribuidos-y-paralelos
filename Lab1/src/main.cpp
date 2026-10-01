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
}