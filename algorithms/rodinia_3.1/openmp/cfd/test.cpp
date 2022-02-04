#include <iostream>

int main() {
#pragma omp target teams distribute parallel for
    for(int i=0; i<1000; i++);

    std::cout << "Hi\n";

    return 0;
}
