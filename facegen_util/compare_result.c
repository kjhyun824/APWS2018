#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <output1.txt> <answer1.txt>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    FILE *f1 = fopen(argv[1], "r");
    if (!f1) {
        fprintf(stderr, "'%s' does not exist.\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    FILE *f2 = fopen(argv[2], "r");
    if (!f2) {
        fprintf(stderr, "'%s' does not exist.\n", argv[2]);
        exit(EXIT_FAILURE);
    }
    int n;
    fscanf(f1, "%d", &n);
    fscanf(f2, "%d", &n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < 64 * 64 * 3; ++j) {
            float a, b;
            fscanf(f1, "%f", &a);
            fscanf(f2, "%f", &b);
            if (fabs(a - b) < 1e-3 || (b != 0 && fabs((a - b) / b) < 1e-3)) {
            } else {
                printf("pixel (%d, %d, %d) of %dth image differs.\n", j / 3 / 64, j / 3 % 64, j % 3, i);
                printf("%s : %f\n", argv[1], a);
                printf("%s : %f\n", argv[2], b);
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("%s and %s are same.\n", argv[1], argv[2]);
    return 0;
}
