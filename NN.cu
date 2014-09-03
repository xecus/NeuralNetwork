#include <iostream>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>
/*
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/optional.hpp>
#include <boost/random.hpp>
*/

#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define	INPUT_LAYER	0
#define	HIDDEN_LAYER	1
#define	OUTPUT_LAYER	2

__device__ double atomicAdd(double* address, double val){
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
	assumed = old;
	old = atomicCAS(address_as_ull, assumed, 
	__double_as_longlong(val + 
	__longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

__global__ void KernelFunc1(double *dA){
	int BlockID = gridDim.x * blockIdx.y + blockIdx.x;
	//int ThreadID = blockDim.x * threadIdx.y + threadIdx.x;
	//int ID = ( blockDim.x * blockDim.y ) * BlockID + ThreadID;

	//OutputVector[i] =  1.0 / ( 1.0 + exp(-OutputVector[i]) );
	dA[ BlockID ] = 1.0 / ( 1.0 + exp(-dA[ BlockID ]) );
	return;
}
__global__ void KernelFunc2(double *dA,double *dB,double *dC){
	int BlockID = gridDim.x * blockIdx.y + blockIdx.x;
	//int ThreadID = blockDim.x * threadIdx.y + threadIdx.x;
	//int ID = ( blockDim.x * blockDim.y ) * BlockID + ThreadID;

	//this->DeltaVector[i] = -( _teach[i] - this->OutputVector[i]) * this->OutputVector[i];
	//this->DeltaVector[i]*= 1.0 - this->OutputVector[i];
	dC[ BlockID ] = -( dB[ BlockID ] - dA[ BlockID ] ) * dA[ BlockID ] * ( 1.0 - dA[ BlockID ] );
	return;
}
__global__ void KernelFunc3(double *dA,double *dB,double *dC,double alpha){
	//int BlockID = gridDim.x * blockIdx.y + blockIdx.x;
	//int ThreadID = blockDim.x * threadIdx.y + threadIdx.x;
	//int ID = ( blockDim.x * blockDim.y ) * BlockID + ThreadID;

	//for(int i=0;i<(this->NumOfNeuron);i++){
	//	for(int j=0;j<(this->PointerOfPrev->NumOfNeuron);j++){
	//		int index = j + i * (this->PointerOfPrev->NumOfNeuron);
	//		this->WeightMatrix[index] += -alpha * this->DeltaVector[i] * this->PointerOfPrev->OutputVector[j];
	//	}
	//}

	//int Imax = gridDim.x;
	//int JMax = gridDim.y;
	int i = blockIdx.x;
	int j = blockIdx.y;
	int index = j + i * gridDim.y; 
	atomicAdd( &dA[index] , -alpha * dC[i] * dB[j] );
	return;
}
__global__ void VarCtrl(double *dA,int index,double val){
	dA[index] = val;
	return;
}

__global__ void KernelFunc5(double *DeltaVector,double *OutputVector){
	int BlockID = gridDim.x * blockIdx.y + blockIdx.x;
	//int ThreadID = blockDim.x * threadIdx.y + threadIdx.x;
	//int ID = ( blockDim.x * blockDim.y ) * BlockID + ThreadID;

	//this->DeltaVector[i] = sum * this->OutputVector[i] * (1.0 - this->OutputVector[i]);
	DeltaVector[BlockID] *= OutputVector[BlockID] * ( 1.0 - OutputVector[BlockID] );
	return;
}

class NeuronLayer{
public:
	cublasHandle_t *cublasHandle;
	cublasStatus_t *cublasStatus;
	int TypeOfLayer;
	int NumOfNeuron;
	NeuronLayer *PointerOfNext;
	NeuronLayer *PointerOfPrev;
	double *WeightMatrix;
	double *OutputVector;
	double *DeltaVector;
	//for CUDA
	double *WeightMatrix_d;
	double *OutputVector_d;
	double *DeltaVector_d;
	double *TeachVector_d;

	bool Mode;

	//Constructer
	NeuronLayer(){
		this->TypeOfLayer = 0;
		this->NumOfNeuron = 0;
		this->PointerOfNext = NULL;
		this->PointerOfPrev = NULL;
		this->WeightMatrix = NULL;
		this->OutputVector = NULL;
		this->DeltaVector = NULL;
		this->WeightMatrix_d = NULL;
		this->OutputVector_d = NULL;
		this->DeltaVector_d = NULL;
		this->TeachVector_d = NULL;
		this->Mode = true;
		return;
	}
	~NeuronLayer(){
		if(this->WeightMatrix) delete[] this->WeightMatrix;
		if(this->OutputVector) delete[] this->OutputVector;
		if(this->DeltaVector) delete[] this->DeltaVector;
		if(this->WeightMatrix_d) checkCudaErrors(cudaFree(this->WeightMatrix_d));
		if(this->OutputVector_d) checkCudaErrors(cudaFree(this->OutputVector_d));
		if(this->DeltaVector_d) checkCudaErrors(cudaFree(this->DeltaVector_d));
		if(this->TeachVector_d) checkCudaErrors(cudaFree(this->TeachVector_d));
		return;	
	}
	void Init(int _type,int _num,NeuronLayer *_pp,NeuronLayer *_np){
		this->TypeOfLayer = _type;
		if(_type != OUTPUT_LAYER) _num+=1;
		this->NumOfNeuron = _num;
		this->PointerOfNext = _np;
		this->PointerOfPrev = _pp;
		this->OutputVector = new double[ _num ];
		for(int i=0;i<_num;i++) OutputVector[i] = 0.0;
		if(this->Mode){
			checkCudaErrors(cudaMalloc((void **)&(this->OutputVector_d), sizeof(double) * _num));
			cudaMemcpy(this->OutputVector_d,this->OutputVector,sizeof(double)*_num,cudaMemcpyHostToDevice);
		}
		if(_type != INPUT_LAYER){
			//Rand from Boost
			/*
			boost::mt19937            gen( static_cast<unsigned long>(time(0)) );
			boost::uniform_smallint<> dst( -10000, 10000 );
			boost::variate_generator< boost::mt19937&, boost::uniform_smallint<> > rand( gen, dst );
			*/
			//Allocate Weight Matrix
			int temp = _num * _pp->NumOfNeuron;
			this->WeightMatrix = new double[ temp ];
			this->DeltaVector = new double[ _num ];
			//for(int i=0;i<temp;i++) this->WeightMatrix[i] = rand() / 10000.0;
			for(int i=0;i<temp;i++) this->WeightMatrix[i] = (double)(rand()%20000-10000)/10000.0;
			for(int i=0;i<_num;i++) this->DeltaVector[i] = 0.0;
			//for CUDA
			if(this->Mode){
				checkCudaErrors(cudaMalloc((void **)&(this->WeightMatrix_d), sizeof(double) * temp));
				cudaMemcpy(this->WeightMatrix_d,this->WeightMatrix,sizeof(double) * temp,cudaMemcpyHostToDevice);
				checkCudaErrors(cudaMalloc((void **)&(this->DeltaVector_d), sizeof(double) * _num));
				cudaMemcpy(this->DeltaVector_d,this->DeltaVector,sizeof(double) * _num ,cudaMemcpyHostToDevice);
			}
		}

		if(_type == OUTPUT_LAYER){
			if(this->Mode){
				checkCudaErrors(cudaMalloc((void **)&(this->TeachVector_d), sizeof(double) * _num));
				cudaMemcpy(this->TeachVector_d,this->OutputVector,sizeof(double)*_num,cudaMemcpyHostToDevice);
			}
		}
		return;
	}

	//Forward Compute
	void Compute(double *_input){
		this->OutputVector[0] = 1.0;
		for(int i=0;i<(this->NumOfNeuron-1);i++) this->OutputVector[i+1] = _input[i];
		cudaMemcpy(this->OutputVector_d,this->OutputVector,sizeof(double)*(this->NumOfNeuron),cudaMemcpyHostToDevice);
		Compute();
	}
	void Compute(){
		if( this->TypeOfLayer != INPUT_LAYER ){
			if(this->Mode){
				for(int i=0;i<(this->NumOfNeuron);i++) this->OutputVector[i] = 0.0;
				int num1 = (this->NumOfNeuron);
				int num2 = (this->PointerOfPrev->NumOfNeuron);
				//cudaMemcpy(this->WeightMatrix_d,this->WeightMatrix,sizeof(double)*num1*num2,cudaMemcpyHostToDevice);
				//cudaMemcpy(this->PointerOfPrev->OutputVector_d,this->PointerOfPrev->OutputVector,sizeof(double)*num2,cudaMemcpyHostToDevice);
				//cudaMemcpy(this->OutputVector_d,this->OutputVector,sizeof(double)*num1,cudaMemcpyHostToDevice);
				double alpha = 1.0 , beta = 0.0;
				cublasDgemv(*(this->cublasHandle),CUBLAS_OP_T,num1,num2,&alpha,this->WeightMatrix_d,num2
					,this->PointerOfPrev->OutputVector_d,1,&beta,this->OutputVector_d,1);
				//cudaMemcpy(this->OutputVector , this->OutputVector_d , sizeof(double)*num1, cudaMemcpyDeviceToHost);
				//cudaMemcpy(this->OutputVector_d,this->OutputVector,sizeof(double)*num1,cudaMemcpyHostToDevice);
				dim3 grid(num1,1),thread(1,1);
				KernelFunc1<<<grid,thread>>>( this->OutputVector_d );
				//cudaMemcpy(this->OutputVector , this->OutputVector_d , sizeof(double)*num1, cudaMemcpyDeviceToHost);
				cudaThreadSynchronize();
				if(this->TypeOfLayer == OUTPUT_LAYER)
				cudaMemcpy(this->OutputVector , this->OutputVector_d , sizeof(double)*num1, cudaMemcpyDeviceToHost);
				if(this->TypeOfLayer != OUTPUT_LAYER){
					dim3 grid(1,1),thread(1,1);
					VarCtrl<<<grid,thread>>>( this->OutputVector_d , 0 ,1.0);
					cudaThreadSynchronize();
				}
			}else{
				for(int i=0;i<(this->NumOfNeuron);i++){
					double temp1 = 0.0;
					for(int j=0;j<(this->NumOfNeuron);j++){
						double temp2 = WeightMatrix[ j + i * this->PointerOfPrev->NumOfNeuron];
						temp2 *= this->PointerOfPrev->OutputVector[j];
						temp1 += temp2;
					}
					OutputVector[i] = temp1;
				}
				for(int i=0;i<(this->NumOfNeuron);i++)
					OutputVector[i] =  1.0 / ( 1.0 + exp(-OutputVector[i]) );	
			}
			if(this->TypeOfLayer != OUTPUT_LAYER) OutputVector[0] = 1.0;
		}
		if( this->TypeOfLayer != OUTPUT_LAYER )
			this->PointerOfNext->Compute();
		return;
	}

	double BackPropagation(double *_teach,double alpha){
		if(this->Mode){
			int num1 = (this->NumOfNeuron);
			int num2 = (this->PointerOfPrev->NumOfNeuron);
			//cudaMemcpy(this->OutputVector_d, this->OutputVector , sizeof(double)*num1, cudaMemcpyHostToDevice);
			cudaMemcpy(this->TeachVector_d, _teach , sizeof(double)*num1, cudaMemcpyHostToDevice);
			//cudaMemcpy(this->DeltaVector_d, this->DeltaVector , sizeof(double)*num1, cudaMemcpyHostToDevice);
			dim3 grid1(num1,1);
			dim3 thread1(1,1);
			KernelFunc2<<<grid1,thread1>>>( this->OutputVector_d , this->TeachVector_d , this->DeltaVector_d );
			cudaThreadSynchronize();
			//cudaMemcpy(this->DeltaVector , this->DeltaVector_d , sizeof(double)*num1, cudaMemcpyDeviceToHost);

			//cudaMemcpy(this->WeightMatrix_d, this->WeightMatrix , sizeof(double)*num1*num2, cudaMemcpyHostToDevice);
			//cudaMemcpy(this->PointerOfPrev->OutputVector_d, this->PointerOfPrev->OutputVector , sizeof(double)*num2, cudaMemcpyHostToDevice);
			//cudaMemcpy(this->DeltaVector_d, this->DeltaVector , sizeof(double)*this->NumOfNeuron, cudaMemcpyHostToDevice);
			dim3 grid2(num1,num2);
			dim3 thread2(1,1);
			KernelFunc3<<<grid2,thread2>>>( this->WeightMatrix_d , this->PointerOfPrev->OutputVector_d , this->DeltaVector_d , alpha );
			cudaThreadSynchronize();
			//cudaMemcpy(this->WeightMatrix , this->WeightMatrix_d , sizeof(double)*num1*num2, cudaMemcpyDeviceToHost);
		}else{
			for(int i=0;i<(this->NumOfNeuron);i++){
				//Parallel
				this->DeltaVector[i] = -( _teach[i] - this->OutputVector[i]) * this->OutputVector[i];
				this->DeltaVector[i]*= 1.0 - this->OutputVector[i];
				for(int j=0;j<(this->PointerOfPrev->NumOfNeuron);j++){
					int index = j + i * (this->PointerOfPrev->NumOfNeuron);
					this->WeightMatrix[index] += -alpha * this->DeltaVector[i] * this->PointerOfPrev->OutputVector[j];
				}
			}
		}
		this->PointerOfPrev->BackPropagation( alpha );
		double e  = 0.0;
		for(int i=0;i<(this->NumOfNeuron);i++) e+= pow(_teach[i] - this->OutputVector[i],2.0);
		e *= 0.5;
		return e;
	}
	void BackPropagation(double alpha){
		if(this->Mode){
			for(int i=0;i<(this->NumOfNeuron);i++) this->DeltaVector[i] = 0.0;
			int num1 = (this->NumOfNeuron);
			int num2 = (this->PointerOfNext->NumOfNeuron);
			//cudaMemcpy(this->PointerOfNext->WeightMatrix_d, this->PointerOfNext->WeightMatrix , sizeof(double)*num1*num2, cudaMemcpyHostToDevice);
			//cudaMemcpy(this->PointerOfNext->DeltaVector_d, this->PointerOfNext->DeltaVector , sizeof(double)*num2, cudaMemcpyHostToDevice);
			//cudaMemcpy(this->DeltaVector_d, this->DeltaVector , sizeof(double)*num1, cudaMemcpyHostToDevice);
			double alpha = 1.0 , beta = 0.0;
			cublasDgemv(*(this->cublasHandle),CUBLAS_OP_N,num1,num2,&alpha,this->PointerOfNext->WeightMatrix_d,num1
				,this->PointerOfNext->DeltaVector_d,1,&beta,this->DeltaVector_d,1);
			//cudaMemcpy(this->DeltaVector, this->DeltaVector_d , sizeof(double)*num1, cudaMemcpyDeviceToHost);

			dim3 grid(num1,1);
			dim3 thread(1,1);
			KernelFunc5<<<grid,thread>>>(this->DeltaVector_d,this->OutputVector_d);
			cudaThreadSynchronize();

			int num3 = (this->NumOfNeuron);
			int num4 = (this->PointerOfPrev->NumOfNeuron);
			//cudaMemcpy(this->WeightMatrix_d,this->WeightMatrix,sizeof(double)*num3*num4,cudaMemcpyHostToDevice);
			//cudaMemcpy(this->PointerOfPrev->OutputVector_d,this->PointerOfPrev->OutputVector,sizeof(double)*num4,cudaMemcpyHostToDevice);
			//cudaMemcpy(this->DeltaVector_d,this->DeltaVector,sizeof(double)*num3,cudaMemcpyHostToDevice);
			dim3 grid2(num3,num4);
			dim3 thread2(1,1);
			KernelFunc3<<<grid2,thread2>>>( this->WeightMatrix_d , this->PointerOfPrev->OutputVector_d , this->DeltaVector_d , alpha );
			cudaThreadSynchronize();
			//cudaMemcpy(this->WeightMatrix , this->WeightMatrix_d , sizeof(double)*num3*num4, cudaMemcpyDeviceToHost);
		}else{
			for(int i=0;i<(this->NumOfNeuron);i++){
				double sum = 0.0;
				for(int j=0;j<(this->PointerOfNext->NumOfNeuron);j++){
					int index = i + j * this->NumOfNeuron;
					sum += this->PointerOfNext->WeightMatrix[index] * this->PointerOfNext->DeltaVector[j];
				}
				this->DeltaVector[i] = sum * this->OutputVector[i] * (1.0 - this->OutputVector[i]);
				for(int j=0;j<(this->PointerOfPrev->NumOfNeuron);j++){
					int index = j + i * (this->PointerOfPrev->NumOfNeuron);
					this->WeightMatrix[index] += -alpha * this->DeltaVector[i] * this->PointerOfPrev->OutputVector[j];
				}
			}
		}
		if(this->PointerOfPrev->TypeOfLayer != INPUT_LAYER) this->PointerOfPrev->BackPropagation( alpha );
		return;
	}

};

class NeuralNetwork{
private:
public:
	cublasHandle_t cublasHandle;
	cublasStatus_t cublasStatus;
	NeuronLayer *layer;
	int NumOfLayer;
	int offset;

	double *Input_d;
	double *Output_d;

	NeuralNetwork(){
		this->layer = NULL;
		this->NumOfLayer = 0;
		this->offset = 0;
		this->cublasHandle = 0;
		this->cublasStatus = cublasCreate(&cublasHandle);
		this->Input_d = NULL;
		this->Output_d = NULL;
	}
	~NeuralNetwork(){
		if(this->Input_d) checkCudaErrors(cudaFree(this->Input_d));
		if(this->Output_d) checkCudaErrors(cudaFree(this->Output_d));
		cublasDestroy(cublasHandle);
	}

	int Init(int num){
		this->NumOfLayer = num;
		layer = new NeuronLayer[num];
		return 0;
	}

	int Add(int num){
		if( this->offset < this->NumOfLayer ){
			std::cout << "(Add) " << this->offset << std::endl;
			if(this->offset==0)
				layer[this->offset].Init( INPUT_LAYER , num , NULL , &layer[this->offset+1]);
			else if(this->offset==this->NumOfLayer-1)
				layer[this->offset].Init( HIDDEN_LAYER , num , &layer[this->offset-1] , &layer[this->offset+1]);
			else 
				layer[this->offset].Init( OUTPUT_LAYER , num , &layer[this->offset-1] , NULL );
			this->offset++;
			return 0;
		}
		std::cout << "(Err) overflow" << std::endl;
		return -1;
	}

	int Add(int id,int type,int num){
		NeuronLayer *p1,*p2;
		p1 = &layer[id-1];
		p2 = &layer[id+1];
		if(id==0) p1 = NULL;
		if(id==this->NumOfLayer-1) p2 = NULL;
		layer[id].Init( type , num , p1 , p2 );
		layer[id].cublasHandle = &(this->cublasHandle);
		layer[id].cublasStatus = &(this->cublasStatus);
		return 0;
	}

	int Show(){
		for(int i=0;i<(this->NumOfLayer);i++){
			if(layer[i].NumOfNeuron!=0){
				std::cout << "[" << i << "/" << this->NumOfLayer << "] (" << &layer[i] << ") Prev=" << layer[i].PointerOfPrev;
				std::cout << " Next=" << layer[i].PointerOfNext << std::endl; 
			}else{
				std::cout << "[" << i << "/" << this->NumOfLayer << "] " << std::endl;
			}
		}
		return 0;
	}

	/*
	void ReadyGPU(int _NumOfSample,double *_Input,double *_Output){
		int DimOfInput = layer[0].NumOfNeuron;
		int DimOfOutput = layer[this->NumOfLayer-1].NumOfNeuron;
		int Size1 = DimOfInput * _NumOfSample;
		int Size2 = DimOfOutput * _NumOfSample;
		checkCudaErrors(cudaMalloc((void **)&(this->Input_d), sizeof(double) * Size1));
		for(int i=0;i<_NumOfSample;i++){
			//Bias
			double TempArray[] = { 1.0 };
			cudaMemcpy(TempArray,this->Input_d+i*DimOfInput,sizeof(double)*1,cudaMemcpyHostToDevice);
			//Input
			cudaMemcpy(_Input+i*(DimOfInput-1),this->Input_d+i*DimOfInput+1,sizeof(double)*(DimOfInput-1),cudaMemcpyHostToDevice);
		}
		checkCudaErrors(cudaMalloc((void **)&(this->Output_d),sizeof(double) * DimOfOutput * _NumOfSample));
		cudaMemcpy(_Output,this->Output_d,sizeof(double)*Size2,cudaMemcpyHostToDevice);
		return;
	}
	*/


	void Compute( double *_input , double *_dest){
		layer[0].Compute( _input );
		if(_dest==NULL) return;
		for(int i=0;i<layer[this->NumOfLayer-1].NumOfNeuron;i++)
		_dest[i] = layer[this->NumOfLayer-1].OutputVector[i];
		return;
	}
	double BackPropagation( double *_teach , double _rate ){
		return layer[this->NumOfLayer-1].BackPropagation( _teach , _rate );
	}

	bool Save(const char* fileName){
		std::vector<double> testVec;
		testVec.push_back( this->NumOfLayer );
		for(int i=1;i<this->NumOfLayer;i++){
			testVec.push_back( layer[i].NumOfNeuron );
			testVec.push_back( layer[i-1].NumOfNeuron );
			cudaMemcpy(layer[i].WeightMatrix, layer[i].WeightMatrix_d,
				sizeof(double)*layer[i].NumOfNeuron*layer[i-1].NumOfNeuron,
			cudaMemcpyDeviceToHost);
			for(int j=0;j<layer[i].NumOfNeuron;j++){
				for(int k=0;k<layer[i-1].NumOfNeuron;k++){
					testVec.push_back( layer[i].WeightMatrix[k+j*layer[i-1].NumOfNeuron] );
				}
			}
		}
		std::ofstream ofs(fileName, std::ios::binary);
		if (ofs.fail()) return false;
		ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(double) * testVec.size());
		if (ofs.fail()) return false;
		ofs.flush();
		if (ofs.fail()) return false;
		return true;	
	}

	bool Load(const char* fileName){
		std::ifstream ifs(fileName, std::ios::binary);
		if (ifs.fail()) return false;
		const size_t fileSize = static_cast<size_t>(ifs.seekg(0, std::ios::end).tellg());
		ifs.seekg(0, std::ios::beg);
		if (fileSize > 0 && fileSize % sizeof(double) == 0){
			std::vector<double> testVec(fileSize / sizeof(double));
			ifs.read(reinterpret_cast<char*>(&testVec[0]), fileSize);
			int offset = 0;
			int LayerNum = testVec[ offset++ ];
			for(int i=1;i<LayerNum;i++){
				int row = testVec[ offset++ ];
				int col = testVec[ offset++ ];
				for(int j=0;j<row;j++) for(int k=0;k<col;k++)
					layer[i].WeightMatrix[k+j*col] = testVec[ offset++ ];
				cudaMemcpy(layer[i].WeightMatrix_d, layer[i].WeightMatrix,
					sizeof(double)*row*col,cudaMemcpyHostToDevice);
			}
		}else return false;
		return true;
	}

};

//Binary
template<typename T_n> bool SaveMatrix(const char* fileName,int _row,int _col,T_n _mat[]){
	std::vector<T_n> testVec;
	testVec.push_back( _row );
	testVec.push_back( _col );
	for(int i=0;i< _row;i++)
	for(int j=0;j< _col;j++)
	testVec.push_back( _mat[ j + i * _col ] );
	std::ofstream ofs(fileName, std::ios::binary);
	if (ofs.fail()) return false;
	ofs.write(reinterpret_cast<const char*>(&testVec[0]), sizeof(T_n) * testVec.size());
	if (ofs.fail()) return false;
	ofs.flush();
	if (ofs.fail()) return false;
	ofs.close();
	if (ofs.fail()) return false;
	return true;
}

//Binary
template<typename T_n> void  LoadCsv(std::string filename,int info[],T_n mat[]){
	std::ifstream ifs( filename.c_str() );
	std::string str;
	if( !ifs ) {
		std::cout << "Error:Input data file not found" << std::endl;
		return;
	}
	int cnt=0;
	int row = 0;
	int col = 0;
	while( std::getline( ifs, str ) ){
		std::string token;
		std::istringstream stream( str );
		int TempCol = 0;
		while( std::getline( stream, token, ',' ) ) {
			std::stringstream ss;
			double temp;
			ss << token;
			ss >> temp;
			mat[cnt] = temp;
			TempCol++;
			cnt++;
		}
		if( col < TempCol ) col = TempCol;
		row++;
	}
	if(info!=NULL){
		info[0] = row;
		info[1] = col;
	}
	return;
}

/*
void MakeData(){

	double Input[] = 
	{1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//1
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	0,0,1,0,0,
	//2
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	//3
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//4
	1,0,0,1,0,
	1,0,0,1,0,
	1,1,1,1,1,
	0,0,0,1,0,
	0,0,0,1,0,
	//5
	1,1,1,1,1,
	1,0,0,0,0,
	1,1,1,1,1,
	0,0,0,0,1,
	1,1,1,1,1,
	//6
	1,0,0,0,0,
	1,0,0,0,0,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//7
	1,1,1,1,1,
	1,0,0,0,1,
	1,0,0,0,1,
	0,0,0,0,1,
	0,0,0,0,1,
	//8
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	//9
	1,1,1,1,1,
	1,0,0,0,1,
	1,1,1,1,1,
	0,0,0,0,1,
	0,0,0,0,1};

	double Output[] = {
	1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,
	0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0
	};

	SaveMatrix<double>("A.bin",10,25,Input);
	SaveMatrix<double>("B.bin",10,10,Output);

	return;
}

*/

class TimeCounter{
private:
public:
	clock_t c0;
	clock_t c1;
	std::string name;
	TimeCounter(){ }
	TimeCounter(std::string _name){ this->name = _name; }

	void start(){
		this->c0 = clock();
	}
	void stop(){
		this->c1 = clock();
	}

	void show(){
		clock_t c = c1 - c0;
		double msec = (static_cast<double>(c) / CLOCKS_PER_SEC) * 1000;
		std::cout << "[" << this->name << "] " << msec << "ms" << std::endl;
	}		

};

int main(int argc,char *argv[]){

	int NumOfSample = 4098;
	int DimOfInput = 105*2;
	int DimOfOutput = 64;
	double *Input = new double[NumOfSample * DimOfInput];
	double *Output = new double[NumOfSample * DimOfOutput];
	int info[2];

	NeuralNetwork nn1;
	nn1.Init(3);
	nn1.Add(0,INPUT_LAYER,DimOfInput);
	nn1.Add(1,HIDDEN_LAYER,200);
	nn1.Add(2,OUTPUT_LAYER,DimOfOutput);
	nn1.Show();
	//MakeData();

	TimeCounter _tc1("File Load");
	TimeCounter _tc2("Trans GPU");
	TimeCounter _tc3("Load NN");
	TimeCounter _tc4("BP");

	_tc1.start();
	//LoadMatrix<double>("A.bin",info,Input);
	LoadCsv<double>("A(Converted).csv",info,Input);
	std::cout << "[A] " << info[0] << " " << info[1] << std::endl;
	//LoadMatrix<double>("B.bin",info,Output);
	LoadCsv<double>("B.csv",info,Output);
	std::cout << "[B] " << info[0] << " " << info[1] << std::endl;
	_tc1.stop();
	_tc1.show();

	for(int i=0;i<(NumOfSample*DimOfOutput);i++)
		Output[i] /= 10.0;

	/*
	_tc2.start();
	nn1.ReadyGPU(NumOfSample,Input,Output);
	_tc2.stop();
	_tc2.show();
	*/

	_tc3.start();
	nn1.Load("N.bin");
	_tc3.stop();
	_tc3.show();

	if( false ){
		std::cout << "Load Passed." << std::endl;
	}else{
		std::cout << "Load Failed." << std::endl;
		int Iter = 0;
		while(1){
			//Back-Propagation
			double e = 0.0;
			_tc3.start();
			for(int i=0;i<NumOfSample;i++){
				nn1.Compute( Input + DimOfInput * i , NULL );
				e+=nn1.BackPropagation( Output + DimOfOutput * i , 0.9 );
			}
			_tc3.stop();
			std::cout << "[" << Iter << "] e=" << e << std::endl;
			_tc3.show();
			//Interval Save Function
			if( Iter!=0 && Iter%100 == 0){
				if( nn1.Save("N.bin") ){
					std::cout << "Interval Save Passed." << std::endl;
				}else{
					std::cout << "Interval Save Failed." << std::endl;
				}
			}
			//Convergence Check
			if( e < 0.001){
				break;
			}
			Iter++;
		}
		if( nn1.Save("N.bin") ){
			std::cout << "Save Passed." << std::endl;
		}else{
			std::cout << "Save Failed." << std::endl;
		}
	}

	double Temp[DimOfOutput];
	for(int i=0;i<NumOfSample;i++){
		nn1.Compute( Input + DimOfInput * i , Temp );
		std::cout << "[" << i << "] ";
		for(int j=0;j<DimOfOutput;j++)
			std::cout << Temp[j] << ",";
		std::cout << std::endl;
	}
	
	delete[] Input;
	delete[] Output;

	return 0;
}
