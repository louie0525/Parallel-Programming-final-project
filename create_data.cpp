#include <iostream>
#include <fstream>
#include <stdlib.h>

using namespace std;

int main(int argc, char **argv){
	if(argc!=2){
		cout << "Input Error!\n";
		cout << "./create_data <NUM_DATA>\n";
		return 1;
	}
	srand( time(NULL) );
	int i, n;
	double K, S; 
	fstream file;
	file.open("data.txt", ios::out);
	sscanf(argv[1],"%d",&n);
	
	file << n << endl;
	
	for(i=0;i<n;i++){
		K = (100.0 - 1.0) * rand() / (RAND_MAX + 1.0) + 1.0;
		S = (100.0 - 1.0) * rand() / (RAND_MAX + 1.0) + 1.0;
		file << K << " " << S << endl;
	}
	
	file.close();
	return 0;
}