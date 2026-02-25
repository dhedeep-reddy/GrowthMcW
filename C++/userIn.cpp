
#include <iostream>
#include <string>
#include <cmath>

using namespace std;

int main(){

    int age;
    string name;
    cout << "Enter ur age\n"<<endl;
    cin >> age;
    cout << age<<endl;

    cin.ignore();

    cout<<"Enter ur name"<<endl;
    getline(cin,name);
    cout << name<< endl;
    

    
    return 0;
}