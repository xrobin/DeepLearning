#include <shared_array_ptr.h>
#include <iostream>
#include <vector>
using std::cout;
using std::endl;
using std::vector;

int main() {
	vector<double> v {1.1, 2.2, 3.3};
	shared_array_ptr<double> sap1(v.begin(), v.end());
	cout << sap1 << endl;
	
	shared_array_ptr<double> sap2(v);
	cout << sap2 << endl;
	
	shared_array_ptr<double> sap3(vector<double> {1.1, 2.2, 3.3});
	cout << sap3 << endl;
}
