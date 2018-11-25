#include <iostream>
#include <armadillo>
#include "libtest.h"
using namespace std;
using namespace arma;

int main(){
    int x = 5;
    int y = 7;
    printf("%d + %d = %d\n", x, y, my_sum(x,y));
    
    int n = 5;
    double my_arr[n];
    for(int i=0; i<n; i++){
        my_arr[i] = i;
    }
    
    //vec u = ones(n);
    vec u = vec(my_arr, n, true, false);
    vec v = 2*ones(n);
    
    vec w = my_vec_sum(u, v); 
    
    cout << w << endl;
    
    double* new_arr = w.memptr();
    for(int i=0; i<n; i++){
        printf("%f\n", new_arr[i]);
    }
    printf("\n");
    
    double xMem[n];
    double yMem[n];
    for(int i=0; i<n; i++){
        xMem[i] = i;
        yMem[i] = i+2;
    }
    
    new_arr = py_my_vec_sum(xMem, yMem, n);
    for(int i=0; i<n; i++){
        printf("%f\n", new_arr[i]);
    }
    printf("\n");
    
    return 0;
}
