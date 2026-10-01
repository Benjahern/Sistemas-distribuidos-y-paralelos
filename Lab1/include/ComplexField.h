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
  
    private:
        int dimM;
        int dimN;
        std::vector<Complex> values;

};