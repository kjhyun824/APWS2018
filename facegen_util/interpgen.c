#include <stdio.h>
#include <stdlib.h>
#include <time.h>
float a[100], b[100];
int main() {
    srand(time(NULL));
    int n;
    scanf("%d", &n);
    printf("%d\n", n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 100; ++j) {
            a[j] = (float)rand() / RAND_MAX * 2 - 1;
            b[j] = (float)rand() / RAND_MAX * 2 - 1;
        }
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < 100; ++k) {
                printf("%.6f ", (a[k] * (n - 1 - j) + b[k] * j) / (n - 1));
            }
            printf("\n");
        }
    }
    return 0;
}
