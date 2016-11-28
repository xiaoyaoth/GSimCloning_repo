#ifndef SOCIAL_FORCE_GPU_H
#define SOCIAL_FORCE_GPU_H

// EXP 3. 8 clone, number of agents 32/256, to see concurrency

#include <vector>
#include <ctime>
#include <Windows.h>
#include <algorithm>
#include <fstream>
#include "inc\helper_math.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <omp.h>

#define USE_GPU 1

using namespace std;

#define DIST(ax, ay, bx, by) sqrt((ax-bx)*(ax-bx)+(ay-by)*(ay-by))

struct obstacleLine
{
	double sx;
	double sy;
	double ex;
	double ey;

	obstacleLine() {}

	__host__ __device__ obstacleLine(double sxx, double syy, double exx, double eyy) {
		sx = sxx;
		sy = syy;
		ex = exx;
		ey = eyy;
	}

	__host__ __device__ void init(double sxx, double syy, double exx, double eyy)
	{
		sx = sxx;
		sy = syy;
		ex = exx;
		ey = eyy;
	}

	__host__ __device__ double pointToLineDist(double2 loc)
	{
		double a, b;
		return this->pointToLineDist(loc, a, b);
	}

	__host__ __device__ double pointToLineDist(double2 loc, double &crx, double &cry)
	{
		double d = DIST(sx, sy, ex, ey);
		double t0 = ((ex - sx) * (loc.x - sx) + (ey - sy) * (loc.y - sy)) / (d * d);

		if (t0 < 0){
			d = sqrt((loc.x - sx) * (loc.x - sx) + (loc.y - sy) * (loc.y - sy));
		}
		else if (t0 > 1){
			d = sqrt((loc.x - ex) * (loc.x - ex) + (loc.y - ey) * (loc.y - ey));
		}
		else{
			d = sqrt(
				(loc.x - (sx + t0 * (ex - sx))) * (loc.x - (sx + t0 * (ex - sx))) +
				(loc.y - (sy + t0 * (ey - sy))) * (loc.y - (sy + t0 * (ey - sy)))
				);
		}
		crx = sx + t0 * (ex - sx);
		cry = sy + t0 * (ey - sy);

		//if (stepCount == 0 && id == 263) {
		//	printf("cross: (%f, %f)\n", crx, cry);
		//	printf("t0: %f, d: %f\n", t0, d);
		//}

		return d;
	}

	__host__ __device__ int intersection2LineSeg(double p0x, double p0y, double p1x, double p1y, double &ix, double &iy)
	{
		double s1x, s1y, s2x, s2y;
		s1x = p1x - p0x;
		s1y = p1y - p0y;
		s2x = ex - sx;
		s2y = ey - sy;

		double s, t;
		s = (-s1y * (p0x - sx) + s1x * (p0y - sy)) / (-s2x * s1y + s1x * s2y);
		t = (s2x * (p0y - sy) - s2y * (p0x - sx)) / (-s2x * s1y + s1x * s2y);

		if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
		{
			// Collision detected
			if (ix != NULL)
				ix = p0x + (t * s1x);
			if (iy != NULL)
				iy = p0y + (t * s1y);
			return 1;
		}
		return 0; // No collision
	}

	__host__ __device__ bool operator == (const obstacleLine & other) {
		return sx == other.sx && sy == other.sy && ex == other.ex && ey == other.ey;
	}

	__host__ __device__ bool operator != (const obstacleLine & other) {
		return !(*this == other);
	}
};
__host__ __device__ inline float dot(const double2& a, const double2& b)
{
	return a.x * b.x + a.y * b.y;
}
__host__ __device__ inline float length(const double2& v)
{
	return sqrtf(dot(v, v));
}
__host__ __device__ inline double2 operator-(const double2& a, const double2& b)
{
	return make_double2(a.x - b.x, a.y - b.y);
}

#define getLastCudaError(msg)	__getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError(const char *errorMessage, const char *file, const int line)
{
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		char message[1024];
		sprintf_s(message, 1024, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
			file, line, errorMessage, (int)err, cudaGetErrorString(err));
		//fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
		//	file, line, errorMessage, (int)err, cudaGetErrorString(err));
		OutputDebugStringA(message);
		system("PAUSE");
		exit(-1);
	}
}

namespace util {
	template<class Type> void hostAllocCopyToDevice(Type *hostPtr, Type **devPtr)//device ptrInWorld must be double star
	{
		size_t size = sizeof(Type);
		cudaMalloc(devPtr, size);
		cudaMemcpy(*devPtr, hostPtr, size, cudaMemcpyHostToDevice);
		getLastCudaError("copyHostToDevice");
	}
}

#define BLOCK_SIZE 32
#define GRID_SIZE(n) (n%BLOCK_SIZE==0 ? n/BLOCK_SIZE : n/BLOCK_SIZE + 1)

/* application related constants */
#define	tao 0.5
#define	A 2000
#define	B 0.1
#define	k1 (1.2 * 100000)
#define k2 (2.4 * 100000) 
#define	maxv 3

#define NUM_CAP 1024
#define NUM_PARAM 3
#define NUM_STEP 500
#define NUM_GOAL 3
#define ENV_DIM 128
#define NUM_CELL 8
#define CELL_DIM 16
#define RADIUS_I 6

#define NUM_WALLS 10

class SocialForceAgent;
class SocialForceClone;

__constant__ int stepCountDev;

typedef struct {
	double2 goal;
	double2 velocity;
	double v0;
	double mass;
	int numNeighbor;
	double2 loc;
} SocialForceAgentData;

class SocialForceAgent {
public:
	int goalIdx;
	uchar4 color;
	int contextId;
	SocialForceClone *myClone;
	SocialForceAgentData data;
	SocialForceAgentData dataCopy;
	double2 goalSeq[NUM_GOAL];

	int flagCloning[NUM_PARAM];
	int flagCloned[NUM_PARAM];
	int cloneIdArray[NUM_PARAM];

	__device__ double correctCrossBoader(double val, double limit);
	__device__ void computeIndivSocialForceRoom(const SocialForceAgentData &myData, const SocialForceAgentData &otherData, double2 &fSum);
	__device__ void computeForceWithWall(const SocialForceAgentData &dataLocal, obstacleLine &wall, const int &cMass, double2 &fSum);
	__device__ void computeWallImpaction(const SocialForceAgentData &dataLocal, obstacleLine &wall, const double2 &newVelo, const double &tick, double &mint);
	__device__ void computeDirection(const SocialForceAgentData &dataLocal, double2 &dvt);
	__device__ void computeSocialForceRoom(SocialForceAgentData &dataLocal, double2 &fSum);
	__device__ void chooseNewGoal(const double2 &newLoc, double epsilon, double2 &newGoal);
	__device__ void step();
	__device__ void init(SocialForceClone* c, int idx);
	__device__ void initNewClone(SocialForceAgent *agent, SocialForceClone *clone);
};

extern "C"
void hookPointerAndData(SocialForceAgent** agentPtrArray, SocialForceAgent* agentArray, int numCap);
extern "C"
void initRootClone(SocialForceClone* cHost, SocialForceClone* cDev);

class AgentPool {
public:
	SocialForceAgent *agentArray;
	SocialForceAgent **agentPtrArray;
	bool *takenFlags;
	//int numElem;

	__host__ AgentPool(int numCap) {
		cudaMalloc((void**)&agentArray, sizeof(SocialForceAgent) * numCap * 1.5);
		cudaMalloc((void**)&agentPtrArray, sizeof(SocialForceAgent*) * numCap);
		cudaMalloc((void**)&takenFlags, sizeof(bool) * numCap);
		cudaMemset(takenFlags, 0, sizeof(bool) * numCap);
		
		hookPointerAndData(agentPtrArray, agentArray, numCap);

		//APUtil::hookPointerAndData << <gSize, BLOCK_SIZE >> >(agentPtrArray, agentArray, numCap);
	}

	__host__ int reorder(int numElem) {
		int l = 0; int r = numElem;
		int i = l, j = l;
		for (; j < r; j++) {
			if (takenFlags[j] == true) {
				swap<SocialForceAgent*>(agentPtrArray, i, j);
				swap<bool>(takenFlags, i, j);
				i++;
			}
		}
		return i;
	}

	template<class T>
	__host__ inline void swap(T * ar, int a, int b) {
		T t1 = ar[a];
		ar[a] = ar[b];
		ar[b] = t1;
	}
};

class SocialForceClone {
public:
	uint numElem;
	curandState *rState;
	AgentPool *ap, *apHost;
	SocialForceAgent **context;
	SocialForceAgent **contextSorted;
	int *cidStarts, *cidEnds;
	bool *cloneFlags;
	int cloneParams[NUM_PARAM];
	obstacleLine walls[NUM_WALLS];
	obstacleLine gates[NUM_PARAM];
	bool takenMap[NUM_CELL * NUM_CELL];
	SocialForceClone *selfDev;
	cudaStream_t myStream;

	uchar4 color;
	int cloneid;
	int parentCloneid;

	int cloneIdArray[NUM_PARAM];
	int cloneMasks[NUM_PARAM];
	int cloneLevel;

	fstream fout;

	__host__ SocialForceClone() {}

	__host__ void init(int id, int pv1[NUM_PARAM], int clondIdArray1[NUM_PARAM]) {
		this->color.x = rand() % 256;
		this->color.y = rand() % 256;
		this->color.z = rand() % 256;
		this->color.w = rand() % 256;
		cloneid = id;
		numElem = 0;
		apHost = new AgentPool(NUM_CAP);
		util::hostAllocCopyToDevice(apHost, &ap);
		cudaMalloc((void**)&rState, sizeof(curandState) * NUM_CAP);
		cudaMalloc((void**)&context, sizeof(SocialForceAgent*) * NUM_CAP);
		cudaMalloc((void**)&contextSorted, sizeof(SocialForceAgent*) * NUM_CAP);
		cudaMalloc((void**)&cloneFlags, sizeof(bool) * NUM_CAP);
		cudaMalloc((void**)&cidStarts, sizeof(int) * NUM_CELL * NUM_CELL);
		cudaMalloc((void**)&cidEnds, sizeof(int) * NUM_CELL * NUM_CELL);
		cudaMemset(context, 0, sizeof(void*) * NUM_CAP);
		cudaMemset(cloneFlags, 0, sizeof(bool) * NUM_CAP);
		
		memcpy(cloneParams, pv1, sizeof(int) * NUM_PARAM);
		memcpy(cloneIdArray, clondIdArray1, sizeof(int) * NUM_PARAM);
		cudaStreamCreate(&myStream);
		
		walls[0].init(0.1 * ENV_DIM, 0.09 * ENV_DIM, 0.1 * ENV_DIM, 0.91 * ENV_DIM);
		walls[1].init(0.09 * ENV_DIM, 0.1 * ENV_DIM, 0.91 * ENV_DIM, 0.1 * ENV_DIM);
		walls[2].init(0.9 * ENV_DIM, 0.09 * ENV_DIM, 0.9 * ENV_DIM, 0.3 * ENV_DIM);
		walls[3].init(0.9 * ENV_DIM, 0.3 * ENV_DIM, 0.9 * ENV_DIM, 0.91 * ENV_DIM);
		walls[4].init(0.09 * ENV_DIM, 0.9 * ENV_DIM, 0.91 * ENV_DIM, 0.9 * ENV_DIM);
		walls[5].init(0.5 * ENV_DIM, 0.7 * ENV_DIM, 0.5 * ENV_DIM, 0.91 * ENV_DIM);
		walls[6].init(0.09 * ENV_DIM, 0.5 * ENV_DIM, 0.3 * ENV_DIM, 0.5 * ENV_DIM);
		walls[7].init(0.5 * ENV_DIM, 0.09 * ENV_DIM, 0.5 * ENV_DIM, 0.3 * ENV_DIM);
		walls[8].init(0.5 * ENV_DIM, 0.3 * ENV_DIM, 0.5 * ENV_DIM, 0.7 * ENV_DIM);
		walls[9].init(0.3 * ENV_DIM, 0.5 * ENV_DIM, 0.91 * ENV_DIM, 0.5 * ENV_DIM);

		walls[9].sx += cloneParams[1]; walls[6].ex -= cloneParams[1];
		walls[7].ey -= cloneParams[2]; walls[8].sy += cloneParams[2];
		walls[8].ey -= cloneParams[0]; walls[5].sy += cloneParams[0];

		gates[0].init(0.5 * ENV_DIM, 0.7 * ENV_DIM - cloneParams[0], 0.5 * ENV_DIM, 0.7 * ENV_DIM + cloneParams[0]);
		gates[1].init(0.3 * ENV_DIM - cloneParams[1], 0.5 * ENV_DIM, 0.3 * ENV_DIM + cloneParams[1], 0.5 * ENV_DIM);
		gates[2].init(0.5 * ENV_DIM, 0.3 * ENV_DIM - cloneParams[2], 0.5 * ENV_DIM, 0.3 * ENV_DIM + cloneParams[2]);

		for (cloneLevel = NUM_PARAM - 1; cloneLevel >= 0; cloneLevel--)
			if (cloneIdArray[cloneLevel] != 0) break;

		for (int i = 0; i < NUM_PARAM; i++) {
			cloneMasks[i] = 1;
			cloneMasks[i] = cloneMasks[i] << cloneIdArray[i];
			cloneMasks[i] = cloneMasks[i] >> 1;
		}

		util::hostAllocCopyToDevice(this, &this->selfDev);
	}
	void step(int stepCount);
	void alterGate(int stepCount);
	void swap();
	void output(int stepCount, char *s) {
		char filename[20];
		sprintf_s(filename, 20, "clone%d_%s.txt", cloneid, s);
		if (stepCount == 1)
			fout.open(filename, fstream::out);
		else
			fout.open(filename, fstream::app);
		fout << "========== stepCount: " << stepCount << " ===========" << endl;
		for (int i = 0; i < NUM_CAP; i++) {
			fout << context[i]->contextId << " [";
			fout << context[i]->data.loc.x << ",";
			fout << context[i]->data.loc.y << "] [";
			fout << context[i]->data.velocity.x << ", ";
			fout << context[i]->data.velocity.y << "] ";
			fout << endl;
		}
		fout.close();
	}
	void output2(int stepCount, char *s) {
		char filename[20];
		sprintf_s(filename, 20, "clone%d_num_%s.txt", cloneid, s);
		if (stepCount == 1)
			fout.open(filename, fstream::out);
		else
			fout.open(filename, fstream::app);
		fout << numElem << endl;
		fout.close();
	}
};

// original tree version
class SocialForceSimApp {
public:
	SocialForceClone **cAll;
	int paintId = 0;
	int totalClone = 8;
	int stepCount = 0;
	int rootCloneId = 0;
	int **cloneTree;

	double2 *debugLocHost, *debugLocDev;
	uchar4 *debugColorHost, *debugColorDev;
	int *debugContextIdHost, *debugContextIdDev;
	int *debugCidStartsHost;
	int	*debugCidEndsHost;

	void performClone(SocialForceClone *parentClone, SocialForceClone *childClone);
	void compareAndEliminate(SocialForceClone *parentClone, SocialForceClone *childClone);
	void proc(int p, int c, bool o, char *s);
	void mst();

	void getLocAndColorFromDevice();
	void initRootClone(SocialForceClone *c, SocialForceClone *cDev);

	int initSimClone() {
		srand(0);

		debugLocHost = new double2[NUM_CAP];
		debugColorHost = new uchar4[NUM_CAP];
		debugContextIdHost = new int[NUM_CAP];
		debugCidStartsHost = new int[NUM_CELL * NUM_CELL];
		debugCidEndsHost = new int[NUM_CELL * NUM_CELL];
		cudaMalloc((void**)&debugLocDev, sizeof(double2) * NUM_CAP);
		cudaMalloc((void**)&debugColorDev, sizeof(uchar4) * NUM_CAP);
		cudaMalloc((void**)&debugContextIdDev, sizeof(int) * NUM_CAP);

		cAll = new SocialForceClone*[totalClone];
		cloneTree = new int*[2];
		int j = 0;

		for (int i = 0; i < totalClone; i++) {
			int cloneParams[NUM_PARAM], cloneIdArray[NUM_PARAM];
			cloneParams[0] = i % 2 + 2;
			cloneParams[1] = (i / 2) % 2 + 2;
			cloneParams[2] = (i / 4) % 2 + 2;
			cloneIdArray[0] = i % 2;
			cloneIdArray[1] = (i / 2) % 2;
			cloneIdArray[2] = (i / 4) % 2;
			wchar_t message[128];
			swprintf_s(message, L"[%d %d%d%d]\t", i, cloneParams[0] - 2, cloneParams[1] - 2, cloneParams[2] - 2);
			OutputDebugString(message);
			cudaMallocHost((void**)&cAll[i], sizeof(SocialForceClone));
			cAll[i]->init(i, cloneParams, cloneIdArray);
		}

		initRootClone(cAll[rootCloneId], cAll[rootCloneId]->selfDev);
		hookPointerAndData(cAll[rootCloneId]->context, cAll[rootCloneId]->apHost->agentArray, NUM_CAP);

		mst();

		return EXIT_SUCCESS;
	}
	void stepApp5(bool o) {
		stepCount++;
		cAll[rootCloneId]->step(stepCount);
		for (int i = 1; i < totalClone; i++)
			proc(cloneTree[0][i], i, 0, "s5");
		for (int j = 0; j < totalClone; j++) {
			cAll[j]->swap();
		}
	}
	void stepApp6(bool o) {
		stepCount++;
		cAll[rootCloneId]->step(stepCount);
		for (int i = 1; i < totalClone; i++)
			proc(i - 1, i, 0, "s6");
		for (int j = 0; j < totalClone; j++) {
			cAll[j]->swap();
		}
	}
	void stepApp(){
		stepCount++;
		cAll[rootCloneId]->step(stepCount);
		
#pragma omp parallel num_threads(3) 
		{
			int tid = omp_get_thread_num();
			switch (tid) {
			case 0: proc(0, 1, 0, "g1"); break;
			case 1: proc(0, 2, 0, "g1"); break;
			case 2: proc(0, 4, 0, "g1"); break;
			}
		}
#pragma omp parallel num_threads(3) 
		{
			int tid = omp_get_thread_num();
			switch (tid) {
			case 0: proc(1, 3, 0, "g1"); break;
			case 1: proc(1, 5, 0, "g1"); break;
			case 2: proc(2, 6, 0, "g1"); break;
			}
		}
#pragma omp parallel num_threads(1) 
		{
			int tid = omp_get_thread_num();
			switch (tid) {
			case 0: proc(3, 7, 0, "g1"); break;
			}
		}


		//cudaDeviceSynchronize();
		printf("\n");
		for (int i = 0; i < totalClone; i++)
			cAll[i]->swap();
		cudaDeviceSynchronize();
		getLastCudaError("step");
	}
};
// all flat version
class SocialForceSimApp2 {
public:
	SocialForceClone **cAll;
	int paintId = 0;
	int totalClone = 8;
	int stepCount = 0;
	int rootCloneId = 0;
	int **cloneTree;

	double2 *debugLocHost, *debugLocDev;
	uchar4 *debugColorHost, *debugColorDev;
	int *debugContextIdHost, *debugContextIdDev;
	int *debugCidStartsHost;
	int	*debugCidEndsHost;

	void performClone(SocialForceClone *parentClone, SocialForceClone *childClone);
	void compareAndEliminate(SocialForceClone *parentClone, SocialForceClone *childClone);
	void proc(int p, int c, bool o, char *s);
	void mst();

	void getLocAndColorFromDevice();
	void initRootClone(SocialForceClone *c, SocialForceClone *cDev);

	int initSimClone() {
		srand(0);

		debugLocHost = new double2[NUM_CAP];
		debugColorHost = new uchar4[NUM_CAP];
		debugContextIdHost = new int[NUM_CAP];
		debugCidStartsHost = new int[NUM_CELL * NUM_CELL];
		debugCidEndsHost = new int[NUM_CELL * NUM_CELL];
		cudaMalloc((void**)&debugLocDev, sizeof(double2) * NUM_CAP);
		cudaMalloc((void**)&debugColorDev, sizeof(uchar4) * NUM_CAP);
		cudaMalloc((void**)&debugContextIdDev, sizeof(int) * NUM_CAP);

		cAll = new SocialForceClone*[totalClone];
		cloneTree = new int*[2];
		int j = 0;

		for (int i = 0; i < totalClone; i++) {
			int cloneParams[NUM_PARAM], cloneIdArray[NUM_PARAM];
			cloneParams[0] = i % 2 + 2;
			cloneParams[1] = (i / 2) % 2 + 2;
			cloneParams[2] = (i / 4) % 2 + 2;
			cloneIdArray[0] = i % 2;
			cloneIdArray[1] = (i / 2) % 2;
			cloneIdArray[2] = (i / 4) % 2;
			wchar_t message[128];
			swprintf_s(message, L"[%d %d%d%d]\t", i, cloneParams[0] - 2, cloneParams[1] - 2, cloneParams[2] - 2);
			OutputDebugString(message);
			cudaMallocHost((void**)&cAll[i], sizeof(SocialForceClone));
			cAll[i]->init(i, cloneParams, cloneIdArray);
		}

		for (int i = 0; i < totalClone; i++) {
			initRootClone(cAll[i], cAll[i]->selfDev);
			hookPointerAndData(cAll[i]->context, cAll[i]->apHost->agentArray, NUM_CAP);
		}

		mst();

		return EXIT_SUCCESS;
	}
	void stepApp(){
		stepCount++;
#pragma omp parallel num_threads(8) 
		{
			int tid = omp_get_thread_num();
			switch (tid) {
			case 0:	cAll[0]->step(stepCount); break;
			case 1:	cAll[1]->step(stepCount); break;
			case 2:	cAll[2]->step(stepCount); break;
			case 3:	cAll[3]->step(stepCount); break;
			case 4:	cAll[4]->step(stepCount); break;
			case 5:	cAll[5]->step(stepCount); break;
			case 6:	cAll[6]->step(stepCount); break;
			case 7:	cAll[7]->step(stepCount); break;
			}
		}

		//cudaDeviceSynchronize();
		printf("\n");
		for (int i = 0; i < totalClone; i++)
			cAll[i]->swap();
		cudaDeviceSynchronize();
		getLastCudaError("step");
	}
};

#endif