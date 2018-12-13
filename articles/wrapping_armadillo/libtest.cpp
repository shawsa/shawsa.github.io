#include <armadillo>

using namespace arma;

extern "C" int my_sum(int x, int y){
    return x+y;
}

vec my_sum(vec x, vec y){
    return x+y;
}

extern "C" void py_my_sum(double* xMem, double* yMem, int len, double retMem[]){
    vec x = vec(xMem, len, true, false);
    vec y = vec(yMem, len, true, false);
    vec ret = my_sum(x, y);
    double* sumMem = ret.memptr();

    /*for(int i=0; i<len; i++){
        //printf("%f\n", ret[i]);
        retMem[i] = ret[i];
    }*/
    memcpy(retMem, sumMem, len*sizeof(double));
}
