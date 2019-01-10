//Parallel Programming Final Project (serial)
//Team: 22
//ver 3.1		2018/12/9	15:00

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <omp.h>


using namespace std;

int NUM_STEPS;
int NUM_DATA;


struct data {
    double K; 
    double S;
    double r;
    double v;
    double T;
	double dt;
	double vdt;
	double u;
	double d;
	double p;
	double* last_C;
	double* last_P;
	double C_price;
	double P_price;
}; 

typedef struct data option_data;

void init(option_data *in){
    in->r = 0.005 ;   // Risk-free rate (5%)
    in->v = 0.3 ;    // Volatility of the underlying (20%)
    in->T = 1.0;   // One year until expiry
	in->dt = in->T/NUM_STEPS;
	in->vdt = in->v*sqrt(in->dt);
	in->u = exp(in->vdt);
	in->d = 1/in->u;
	in->p = (exp(in->r*in->dt)-in->d)/(in->u-in->d);
	in->last_C = new double [NUM_STEPS+1];
	in->last_P = new double [NUM_STEPS+1];
}

//Call Option 認購期權
double CallOption(const double& S,const double& K,const double& vDt,const int& i){
	double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - K;      
	return max(d,(double)0); 
}

//Put Option 認沽期權
double PutOption(const double& S,const double& K,const double& vDt,const int& i){
	double d = K-S * exp(vDt * (2.0 * i - NUM_STEPS));          
	return max(d,(double)0); 
}

void last_step_price(option_data* temp_data,int NUM_STEPS){
	for(int k = 0; k <=NUM_STEPS; k++){
		double sd = temp_data->S * exp(temp_data->vdt * (2.0 * (NUM_STEPS-k) - NUM_STEPS)) ; 
		
		temp_data->last_C[k] = max(sd-temp_data->K,(double)0);
		temp_data->last_P[k] = max(temp_data->K-sd,(double)0);
	}
	
	return;
}

void trace_back(option_data* temp_option){
	double* temp_C = new double [NUM_STEPS+1];
	double* temp_P = new double [NUM_STEPS+1];
	// double Newtemp_C[NUM_STEPS+1];
	// double Newtemp_P[NUM_STEPS+1];

	// #pragma omp parallel for
	for(int i =0;i<=NUM_STEPS;i++){
		temp_C[i] = temp_option->last_C[i];
		temp_P[i] = temp_option->last_P[i];
	}
	
	for(int t = 0; t < NUM_STEPS; ++t){
		#pragma omp parallel for firstprivate(t)
		for(int i = 0; i < NUM_STEPS-t; ++i){
			temp_C[i] = exp(-temp_option->r*temp_option->dt) * (temp_option->p*temp_C[i] + (1-temp_option->p)*temp_C[i+1]);
			temp_P[i] = exp(-temp_option->r*temp_option->dt) * (temp_option->p*temp_P[i] + (1-temp_option->p)*temp_P[i+1]);
		}
	}
	temp_option->C_price = temp_C[0];
	temp_option->P_price = temp_P[0];
	delete [] temp_C;
	delete [] temp_P;
	
	return;
}

int main(int argc, char **argv){
	if(argc!=3){
		cout << "Input error!\n";
		cout << "./project <NUM_STEPS> <DATA_FILE>\n";
		return 1;
	}
	sscanf(argv[1],"%d",&NUM_STEPS);
	ifstream infile(argv[2]);
	if(!infile) {
		cout << "Can not open input file!\n";
		return 1;
	}
	
	infile >> NUM_DATA;
	cout << "num_data = " << NUM_DATA << endl;
	cout << "num_step = " << NUM_STEPS << endl;
    
	vector<option_data*> option_vector;
	
	for(int i =0;i<NUM_DATA;i++){
		double K, S; 
		infile >> K >> S;
	    option_data *temp_data = new option_data();
		init(temp_data);
		temp_data->K = K;
		temp_data->S = S;
		
		last_step_price(temp_data,NUM_STEPS);
        option_vector.push_back(temp_data);
	}
    
	//trace back
	#pragma omp parallel for
	for(int i =0;i<NUM_DATA;i++){
		trace_back(option_vector[i]);
	}
	
	//print
	for(int i =0;i<NUM_DATA;i++){
	    option_data* print_price = option_vector[i];
		printf("Data %d: Call Price: %.5f\tPut Price: %.5f\n", i, print_price->C_price, print_price->P_price);
	}
	
	return 0;
}
