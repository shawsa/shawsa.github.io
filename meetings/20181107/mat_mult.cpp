#include <armadillo>
#include <chrono>
using namespace arma;

int main(){
    const int n = 5000;
    arma::arma_rng::set_seed_random();
    
    mat A = randu<mat>(n,n);

    auto start = std::chrono::high_resolution_clock::now();

    mat B = A*A;

    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = finish - start;
    printf("%f s\n", elapsed.count());

    return 0;
}
