# include "../include/ComplexField.h"
# include <stdexcept>


ComplexField::ComplexField(int M, int N) : dimM(M),dimN(N) {
    if(!isPowerOf2(M) || !isPowerOf2(N)){
        throw std::invalid_argument("M y N deben ser potencias de 2");
    }
    values.assign(static_cast<size_t>(M)*N, Complex(0.0,0.0));
}

ComplexField::Complex& ComplexField::at(int m, int n){
    return values[static_cast<size_t>(m)*dimN+n];
}

const ComplexField::Complex& ComplexField::at(int m, int n) const{
    return values[static_cast<size_t>(m)*dimN+n];
}

bool ComplexField::isPowerOf2(int x){
    return x > 0 && (x & (x-1)) == 0;
}

int ComplexField::bitReverse(int x, int bits){
    int reversed = 0;
    while(bits>0){
        reversed = (reversed << 1) | (x & 1);
        x >>= 1;
        bits--;
    }
    return reversed;
}

void ComplexField::bitReversePermute(Complex* arr, int n){
    int bits = 0;
    int tam = 1;
    while(tam < n){
        tam*=2;
        bits++;
    }

    int i = 0;
    while(i<n){
        int j = bitReverse(i,bits);
        if(i<j){
            std::swap(arr[i],arr[j]);
        }
        i++;
    }
}


void ComplexField::getRow(int m,Complex* row) const{
    int n = 0;
    while(n < dimN){
        row[n] = at(m,n);
        n++;
    }
}

void ComplexField::setRow(int m, const Complex* row){
    int n = 0;
    while(n < dimN){
        at(m,n) =row[n];
        n++;
    }
}

void ComplexField::getCol(int n,Complex* col) const{
    int m = 0;
    while(m < dimM){
        col[m] = at(m,n);
        m++;
    }
}

void ComplexField::setCol(int n, const Complex* col){
    int m = 0;
    while(m < dimM){
        at(m,n) = col[m];
        m++;
    }
}