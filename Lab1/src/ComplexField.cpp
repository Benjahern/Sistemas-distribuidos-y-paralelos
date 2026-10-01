# include "../include/ComplexField.h"
# include <cmath>
# include <random>
# include <utility>

ComplexField::ComplexField(size_t M, size_t N)
    : rows_(M), cols_(N), data_(M * N, Complex(0.0, 0.0)) {}

ComplexField::ComplexField(size_t M, size_t N, const Complex& initial_value)
    : rows_(M), cols_(N), data_(M * N, initial_value) {}

bool ComplexField::isPowerOf2(size_t x){
    return x != 0 && (x & (x - 1)) == 0;
}

size_t ComplexField::bitReverse(size_t x, size_t bits){
    size_t reversed = 0;
    while(bits > 0){
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
        bits--;
    }
    return reversed;
}

void ComplexField::bitReversePermute(Complex* arr, size_t n){
    size_t bits = 0;
    size_t tam = 1;
    while(tam < n){
        tam *= 2;
        bits++;
    }
    size_t i = 0;
    while(i < n){
        size_t j = bitReverse(i, bits);
        if(i < j){
            std::swap(arr[i], arr[j]);
        }
        i++;
    }
}

void ComplexField::getRow(size_t m, Complex* row) const{
    size_t n = 0;
    while(n < cols_){
        row[n] = at(m, n);
        n++;
    }
}

void ComplexField::setRow(size_t m, const Complex* row){
    size_t n = 0;
    while(n < cols_){
        at(m, n) = row[n];
        n++;
    }
}

void ComplexField::getCol(size_t n, Complex* col) const{
    size_t m = 0;
    while(m < rows_){
        col[m] = at(m, n);
        m++;
    }
}

void ComplexField::setCol(size_t n, const Complex* col){
    size_t m = 0;
    while(m < rows_){
        at(m, n) = col[m];
        m++;
    }
}

void ComplexField::fillImpulse(){
    size_t m = 0;
    while(m < rows_){
        size_t n = 0;
        while(n < cols_){
            at(m, n) = Complex(0.0, 0.0);
            n++;
        }
        m++;
    }
    at(0, 0) = Complex(1.0, 0.0);
}

void ComplexField::fillSine(size_t k, size_t l){
    const double pi = std::acos(-1.0);
    size_t m = 0;
    while(m < rows_){
        size_t n = 0;
        while(n < cols_){
            double fase = 2.0 * pi * (double(k) * m / rows_ + double(l) * n / cols_);
            at(m, n) = Complex(std::sin(fase), 0.0);
            n++;
        }
        m++;
    }
}

void ComplexField::fillRandom(unsigned seed){
    std::mt19937_64 generador(seed);
    std::uniform_real_distribution<double> distribucion(-1.0, 1.0);
    size_t m = 0;
    while(m < rows_){
        size_t n = 0;
        while(n < cols_){
            double real = distribucion(generador);
            double imag = distribucion(generador);
            at(m, n) = Complex(real, imag);
            n++;
        }
        m++;
    }
}