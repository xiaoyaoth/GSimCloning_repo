#include "gsimcore.cuh"
//#include "boid.cuh"
#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif
//#include "test.cuh"
//#include "socialForce.cuh"
#include "SimpleClone.cuh"
int main(int argc, char *argv[]){
	//argv[1]: config.txt
	//argv[2]: numAgent
	init<SimpleAgentData>(argv[1]);
	SimpleModel *model_h = new SimpleModel();
	/*Main work started here*/

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	doLoop(model_h);

	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("execution time: %f\n", time);
}
