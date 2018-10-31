#include <iostream>
#include "hello_world.hpp"

int main(){
  std::cout << "Hello World from CPU!" << std::endl;
  launch_cuda();
  return 0;
}
