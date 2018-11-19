#define ARMA_DONT_PRINT_ERRORS
#include <armadillo>
#include <chrono>
using namespace arma;

int main(){

    const int runs = 10;
    double times[runs];

    const int n = 1000;
    const int m = 10000;

    const double epsilon = 45.0;

    const double PI = datum::pi;

    arma::arma_rng::set_seed_random();
    
    vec x = linspace<vec>(0, 2*PI, n);
    vec z = linspace<vec>(0, 2*PI, m);
    vec y = sin(x);

    mat A;
    vec c, u;

    for(int r=0; r<runs; r++){
        auto start = std::chrono::high_resolution_clock::now();

        A = repmat(x, 1, n) - repmat(x.t(), n, 1);
        A = exp( -pow( epsilon*abs(A) , 2) );
        c = solve(A, y);
        A = repmat(z, 1, n) - repmat(x.t(), m, 1);
        A = exp( -pow( epsilon*abs(A) , 2) );
        u = A*c;

        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        times[r] = elapsed.count();
    }
    u.save("u_sine.dat", raw_ascii);

    double best = times[0];
    for(int r=1; r<runs; r++){
        if(times[r] < best){
            best = times[r];
        }
    }
    printf("Best of %d runs: \t%fs\n", runs, best);


    return 0;
}
