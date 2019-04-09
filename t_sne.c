#include <stdio.h>
#include "mnist.h"


int main()
{
    load_mnist();
    int cnt = 0;
    for (int i=0; i<784; i++) {
		
        printf("%1.1f ", train_image[0][i]);
		if ((i+1) % 28 == 0) putchar('\n');
	}
}