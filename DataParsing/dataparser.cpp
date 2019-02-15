#include<iostream>
#include<string>
#include"Runners/gloveRunner.h"
#include"Runners/siftRunner.h"

using namespace std;

void writeError(){
    cout << "Could not find any parser for the specified file - try one of the following parsers" << endl;
    cout << "glove | sift" << endl;
}

int main(int argc, char** args){
    if(argc == 1){
        writeError();
        return 0;
    }
    string parser = args[1];

    if(parser == "glove"){
        GloveRunner runner;
        runner.run(argc, args);
    } else if(parser == "sift"){
        SiftRunner runner;
        runner.run(argc, args);
    } else {
        writeError();
    }

    return 0;
}