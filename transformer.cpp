#include <cstdio>

int main() {
	FILE *fin = fopen("lcsInitialWindows.txt", "r");
	FILE *fout = fopen("lcsInitialLocationsVisual.txt", "w");
	int a, b;
	while (fscanf(fin, "%*lf %*lf %*lf %d %d", &a, &b) == 2)
		fprintf(fout, "%d %d\n", a, b);
	fclose(fin);
	fclose(fout);
	return 0;
}
