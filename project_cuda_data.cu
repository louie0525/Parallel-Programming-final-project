//Parallel Programming Final Project (CUDA)
//Team: 22
//ver 1.2		2018/12/16	16:20

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <time.h>
#include <cuda.h>


using namespace std;

int NUM_STEPS;
int NUM_DATA;
double 	*C_gpu,
		*P_gpu,
		*C_price_gpu,
		*P_price_gpu;

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



__global__ void trace_back_gpu(double* C_option, double* P_option, double* C_price, double* P_price, double r, double dt, double p, int NUM_DATA, int NUM_STEPS){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x<NUM_DATA){
		for(int t = 0; t < NUM_STEPS; ++t){
			for(int i = 0; i < NUM_STEPS-t; ++i){
				C_option[x*(NUM_STEPS+1)+i] = exp(-r*dt) * (p*C_option[x*(NUM_STEPS+1)+i] + (1-p)*C_option[x*(NUM_STEPS+1)+i+1]);
				P_option[x*(NUM_STEPS+1)+i] = exp(-r*dt) * (p*P_option[x*(NUM_STEPS+1)+i] + (1-p)*P_option[x*(NUM_STEPS+1)+i+1]);
			}
		}			
	}

	C_price[x] = C_option[x*(NUM_STEPS+1)];	
	P_price[x] = P_option[x*(NUM_STEPS+1)];

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
    
	size_t size = ((NUM_STEPS+1)*NUM_DATA)*sizeof(double);
	vector<option_data*> option_vector;
	
	for(int i = 0; i < NUM_DATA; ++i){
		double K, S; 
		infile >> K >> S;
	    option_data *temp_data = new option_data();
		init(temp_data);
		temp_data->K = K;
		temp_data->S = S;
		
		last_step_price(temp_data,NUM_STEPS);
        option_vector.push_back(temp_data);
	}

	double** C_matrix = new double*[NUM_DATA];
	double** P_matrix = new double*[NUM_DATA];
	double* C_array = new double[(NUM_STEPS+1)*NUM_DATA];
	double* P_array = new double[(NUM_STEPS+1)*NUM_DATA];
	
	for(int i = 0; i < NUM_DATA; ++i){
		C_matrix[i] = option_vector[i]->last_C;
		P_matrix[i] = option_vector[i]->last_P;
		for(int j = 0; j < NUM_STEPS+1; ++j){
			C_array[i*(NUM_STEPS+1)+j] = C_matrix[i][j];
			P_array[i*(NUM_STEPS+1)+j] = P_matrix[i][j];
		}
	}

	double r = option_vector[0]->r;
	double dt = option_vector[0]->dt;
	double p = option_vector[0]->p;
	double C_price[NUM_DATA];
	double P_price[NUM_DATA];


    
	cudaMalloc((void**)&C_gpu, size);
	cudaMalloc((void**)&P_gpu, size);
	cudaMalloc((void**)&C_price_gpu, NUM_DATA*sizeof(double));
	cudaMalloc((void**)&P_price_gpu, NUM_DATA*sizeof(double));
	cudaMemcpy(C_gpu, C_array, size, cudaMemcpyHostToDevice);
	cudaMemcpy(P_gpu, P_array, size, cudaMemcpyHostToDevice);
	//trace_back_gpu<<<NUM_DATA/32 + 1,32>>>(C_gpu, C_price_gpu, r, dt, p, NUM_DATA, NUM_STEPS);
	trace_back_gpu<<<NUM_DATA/32 + 1,32>>>(C_gpu, P_gpu, C_price_gpu, P_price_gpu, r, dt, p, NUM_DATA, NUM_STEPS);
	cudaMemcpy(C_price, C_price_gpu, NUM_DATA*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(P_price, P_price_gpu, NUM_DATA*sizeof(double), cudaMemcpyDeviceToHost);
	
	//print
	for(int i =0;i<NUM_DATA;i++){
	    option_data* print_price = option_vector[i];
		printf("Data %d: Call Price: %.5f\tPut Price: %.5f\n", i, C_price[i], P_price[i]);
	}
	
	return 0;
}
