#include <iostream>
#include <experimental/filesystem>
#include "SpeedupTests.h"


int main() {

    std::string input = "../input";
    std::string output = "../output";

    if(std::experimental::filesystem::exists(output + "/SpeedUpResults.csv")){
        std::experimental::filesystem::remove(output + "/SpeedUpResults.csv");
    }

    goWithDifferentNumberOfPoints(input , output);

    return 0;
}
