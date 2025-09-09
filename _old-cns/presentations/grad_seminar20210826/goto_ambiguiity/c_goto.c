#include <stdio.h>
int main(){
int a, b;
a = 5;
goto *(&&LABEL + 7);
LABEL:
a = 1; 
b = a;
printf("%d\n", b);
int* testLabel;
TEST: testLabel = &&TEST;
//TEST:printf("%ld\n", &&TEST - &&LABEL);
}
