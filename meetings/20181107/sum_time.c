# include <stdio.h>
# include <time.h>
#include <stdlib.h>

int main(){

int n = 1000000;
double lst[n];
for(int i=0; i<n; i++){
    lst[i] = rand();
}

srand(time(0));

clock_t begin = clock();
double sum = 0;
for(int i=0; i<n; i++){
    sum += lst[i];
}
clock_t end = clock();

double time_spent = (double)(end - begin) *1000/ CLOCKS_PER_SEC;

printf("%fms\n", time_spent);

return(0);
}
