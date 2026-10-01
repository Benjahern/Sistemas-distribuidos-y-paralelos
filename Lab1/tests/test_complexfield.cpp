# include "ComplexField.h"
# include <cassert>
# include <iostream>
# include <cstdio>
# include <stdexcept>

using Complex = ComplexField::Complex;

int main(){

    // Los indices de 3 bits invertidos deben dar 0 4 2 6 1 5 3 7
    size_t esperado[8] = {0, 4, 2, 6, 1, 5, 3, 7};
    size_t i = 0;
    while(i < 8){
        assert(ComplexField::bitReverse(i, 3) == esperado[i]);
        i++;
    }

    // Permuta un arreglo 0..7 y debe quedar en el mismo orden de esperado
    Complex arr[8];
    i = 0;
    while(i < 8){
        arr[i] = Complex(i, 0.0);
        i++;
    }
    ComplexField::bitReversePermute(arr, 8);
    i = 0;
    while(i < 8){
        assert(arr[i].real() == esperado[i]);
        i++;
    }

    // Grilla de 8 filas y 4 columnas, cada celda vale 10*m + n
    ComplexField g(8, 4);
    size_t m = 0;
    while(m < 8){
        size_t n = 0;
        while(n < 4){
            g.at(m, n) = Complex(10.0 * m + n, 0.0);
            n++;
        }
        m++;
    }

    // Saca la fila 2 y debe ser 20 21 22 23 (fila tiene 4 elementos)
    Complex fila[4];
    g.getRow(2, fila);
    i = 0;
    while(i < 4){
        assert(fila[i].real() == 20 + i);
        i++;
    }

    // Saca la columna 1 y debe ser 1 11 21 ... 71 (col tiene 8 elementos)
    Complex col[8];
    g.getCol(1, col);
    i = 0;
    while(i < 8){
        assert(col[i].real() == 10 * i + 1);
        i++;
    }

    // Pone una columna nueva en la 3 y al leerla debe ser la misma
    Complex nueva[8];
    i = 0;
    while(i < 8){
        nueva[i] = Complex(100.0 + i, 0.0);
        i++;
    }
    g.setCol(3, nueva);
    g.getCol(3, col);
    i = 0;
    while(i < 8){
        assert(col[i] == nueva[i]);
        i++;
    }

    // Pone una fila nueva en la 5 y al leerla debe ser la misma
    Complex otra[4];
    i = 0;
    while(i < 4){
        otra[i] = Complex(500.0 + i, 1.0);
        i++;
    }
    g.setRow(5, otra);
    g.getRow(5, fila);
    i = 0;
    while(i < 4){
        assert(fila[i] == otra[i]);
        i++;
    }

    // row(m) apunta directo a la fila, sin copiar
    assert(g.row(2)[1] == g.at(2, 1));

    // Constructor por defecto da grilla vacia y el otro la llena con un valor
    ComplexField vacio;
    assert(vacio.rows() == 0);
    assert(vacio.cols() == 0);
    ComplexField lleno(2, 2, Complex(1.0, 1.0));
    assert(lleno.at(1, 1) == Complex(1.0, 1.0));
    assert(lleno.size() == 4);

    // Con la misma semilla salen los mismos datos, con otra semilla distintos
    ComplexField r1(8, 4);
    ComplexField r2(8, 4);
    ComplexField r3(8, 4);
    r1.fillRandom(42);
    r2.fillRandom(42);
    r3.fillRandom(43);
    bool iguales = true;
    bool distintos = false;
    m = 0;
    while(m < 8){
        size_t n = 0;
        while(n < 4){
            if(r1.at(m, n) != r2.at(m, n)){
                iguales = false;
            }
            if(r1.at(m, n) != r3.at(m, n)){
                distintos = true;
            }
            n++;
        }
        m++;
    }
    assert(iguales);
    assert(distintos);

    // Guarda r1 en un archivo y al leerlo debe quedar exactamente igual
    r1.writeDat("/tmp/test_complexfield.dat");
    ComplexField leido;
    leido.readDat("/tmp/test_complexfield.dat");
    assert(leido.rows() == 8);
    assert(leido.cols() == 4);
    bool identicos = true;
    m = 0;
    while(m < 8){
        size_t n = 0;
        while(n < 4){
            if(leido.at(m, n) != r1.at(m, n)){
                identicos = false;
            }
            n++;
        }
        m++;
    }
    assert(identicos);
    std::remove("/tmp/test_complexfield.dat");

    // Leer un archivo que no existe debe lanzar error
    bool lanzo = false;
    try{
        leido.readDat("/tmp/no_existe_nunca.dat");
    }catch(const std::runtime_error&){
        lanzo = true;
    }
    assert(lanzo);

    // El impulso tiene un 1 en (0,0) y ceros en el resto
    ComplexField imp(4, 4);
    imp.fillImpulse();
    assert(imp.at(0, 0) == Complex(1.0, 0.0));
    assert(imp.at(1, 2) == Complex(0.0, 0.0));

    // En el seno la fase de (0,0) es 0, asi que ahi vale 0
    ComplexField sen(8, 8);
    sen.fillSine(2, 3);
    assert(sen.at(0, 0) == Complex(0.0, 0.0));

    // El 1 y el 8 son potencias de 2, el 0 y el 6 no
    assert(ComplexField::isPowerOf2(1));
    assert(ComplexField::isPowerOf2(8));
    assert(!ComplexField::isPowerOf2(0));
    assert(!ComplexField::isPowerOf2(6));

    std::cout << "ComplexField: TODAS LAS PRUEBAS PASARON\n";
    return 0;
}