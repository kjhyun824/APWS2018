#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));
    int n;
    scanf("%d", &n);
    printf("%d\n", n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 100; ++j) {
            printf("%.6f ", (float)rand() / RAND_MAX * 2 - 1);
        }
        printf("\n");
    }
    return 0;
}
