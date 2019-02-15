#include<iostream>
#include<string>
#include"Runners/gloveRunner.h"
#include"Runners/siftRunner.h"

using namespace std;

int main(int argc, char** args){

    string parser = args[1];

    if(parser == "glove"){
        GloveRunner runner;
        runner.run(argc, args);
    } else if(parser == "sift"){
        SiftRunner runner;
        runner.run(argc, args);
    } else {
        cout << "Could not find any parser for the specified file - try one of the following parsers" << endl;
        cout << "glove | sift" << endl;
    }

    return 0;
}