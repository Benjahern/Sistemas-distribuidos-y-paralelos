# pragma once
# include <vector>
# include <complex>

class ComplexField{

    public: 
        using Complex = std::complex<double>;

        ComplexField(int M, int N);

        int rows() const {
            return dimM;
        }
        int cols() const {
            return dimN;
        }

        Complex& at(int m, int n);
        const Complex& at(int m, int n) const;

        Complex* data(){
            return values.data();
        }
        const Complex* data() const{
            return values.data();
        }

        static bool isPowerOf2(int x);

        static int bitReverse(int x, int bits);

        static void bitReversePermute(Complex* arr, int n);

        // Saca la fila m de la grilla y la deja en row (row tiene cols() elementos)
        void getRow(int m, Complex* row) const;

        // Toma row y la pone en la fila m de la grilla
        void setRow(int m, const Complex* row);
  
        // Saca la columna n de la grilla y la deja en col (col tiene rows() elementos)
        void getCol(int n, Complex* col) const;

        // Toma col y la pone en la columna n de la grilla
        void setCol(int n, const Complex* col);

        void fillImpulse();

        void fillSine(int k, int l);

        void fillRandom(unsigned seed);

    private:
        int dimM;
        int dimN;
        std::vector<Complex> values;

};