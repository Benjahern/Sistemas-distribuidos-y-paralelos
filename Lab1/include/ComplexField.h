# pragma once
# include <vector>
# include <complex>
# include <cstddef>
# include <string>

class ComplexField{

    public: 
        using Complex = std::complex<double>;

        ComplexField() = default;

        ComplexField(size_t M, size_t N);

        ComplexField(size_t M, size_t N, const Complex& initial_value);

        size_t rows() const {
            return rows_;
        }
        size_t cols() const {
            return cols_;
        }

        size_t size() const {
            return data_.size();
        }

        Complex& at(size_t m, size_t n){
            return data_[m * cols_ + n];
        }

        const Complex& at(size_t m, size_t n) const{
            return data_[m * cols_ + n];
        }

        Complex& operator()(size_t m, size_t n){
            return data_[m * cols_ + n];
        }

        const Complex& operator()(size_t m, size_t n) const{
            return data_[m * cols_ + n];
        }

        Complex* data(){
            return data_.data();
        }
        const Complex* data() const{
            return data_.data();
        }

        Complex* row(size_t m){
            return data_.data() + m * cols_;
        }

        const Complex* row(size_t m) const{
            return data_.data() + m * cols_;
        }

        static bool isPowerOf2(size_t x);

        static size_t bitReverse(size_t x, size_t bits);

        static void bitReversePermute(Complex* arr, size_t n);

        // Saca la fila m de la grilla y la deja en fila (fila tiene cols() elementos)
        void getRow(size_t m, Complex* fila) const;

        // Toma fila y la pone en la fila m de la grilla
        void setRow(size_t m, const Complex* fila);
  
        // Saca la columna n de la grilla y la deja en columna (columna tiene rows() elementos)
        void getCol(size_t n, Complex* columna) const;

        // Toma columna y la pone en la columna n de la grilla
        void setCol(size_t n, const Complex* columna);

        void fillImpulse();

        void fillSine(size_t k, size_t l);

        void fillRandom(unsigned seed);

        void writeDat(const std::string& path) const;

        void readDat(const std::string& path);

    private:
        size_t rows_{0};
        size_t cols_{0};
        std::vector<Complex> data_;

};