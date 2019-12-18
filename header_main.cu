#include <stdio.h>
#include "header.h"  /* Include the header here, to obtain the function declaration */

int main(void)
{
    int y = foo1(3);  /* Use the function here */
    printf("%d\n", y);
    y = foo2(3);
    printf("%d\n", y);
    return 0;
}
