#include "cuda_runtime.h"
#include <fstream>
#include "SocialForceGPU.h"
#include <omp.h>
__global__ void testFunc() {

}

namespace NeighborModule {
	__device__ int zcode(int x, int y) {
		//return x * NUM_CELL + y;
		x &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
		y &= 0x0000ffff;					// x = ---- ---- ---- ---- fedc ba98 7654 3210
		x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		y = (y ^ (y << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
		y = (y ^ (y << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
		y = (y ^ (y << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
		y = (y ^ (y << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
		x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
		return x | (y << 1);
	}

	__device__ int zcode(const double2 &loc) {
		int ix = loc.x / (ENV_DIM / NUM_CELL);
		int iy = loc.y / (ENV_DIM / NUM_CELL);
		return zcode(ix, iy);
	}

	__device__ int zcode(SocialForceAgent *agent) {
		return zcode(agent->data.loc);
	}

	__device__ void swap(SocialForceAgent** agentPtrs, int a, int b) {
		SocialForceAgent* temp = agentPtrs[a];
		agentPtrs[a] = agentPtrs[b];
		agentPtrs[b] = temp;
	}

	__device__ void quickSortByAgentLoc(SocialForceAgent** agentPtrs, curandState &rState, int l, int r) {
		if (l == r)
			return;
		int pi = l + curand(&rState) % (r - l);
		swap(agentPtrs, l, pi);
		SocialForceAgent* pivot = agentPtrs[l];

		int i = l + 1, j = l + 1;
		for (; j < r; j++) {
			if (zcode(agentPtrs[j]) < zcode(pivot)) {
				swap(agentPtrs, i, j);
				i++;
			}
		}
		swap(agentPtrs, l, i - 1);
		quickSortByAgentLoc(agentPtrs, rState, l, i - 1);
		quickSortByAgentLoc(agentPtrs, rState, i, r);
	}

	__global__ void sortAgentByLocKernel(SocialForceAgent** agentPtrsToSort, curandState *rState, int numCap) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		curandState &rStateLocal = *rState;
		if (idx == 0)
			quickSortByAgentLoc(agentPtrsToSort, rStateLocal, 0, numCap);
	}

	__global__ void setCidStartEndKernel(SocialForceAgent** contextSorted, int* cidStarts, int* cidEnds, int numCap) {
		const int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numCap && idx > 0) {
			int cid = zcode(contextSorted[idx]);
			int cidPrev = zcode(contextSorted[idx - 1]);
			if (cid != cidPrev) {
				cidStarts[cid] = idx;
				cidEnds[cidPrev] = idx;
			}
		}
		if (idx == 0) {
			int cid = zcode(contextSorted[0]);
			cidStarts[cid] = 0;

			cid = zcode(contextSorted[numCap - 1]);
			cidEnds[cid] = numCap;
		}
	}
}

extern "C"
void runTest() {
	testFunc << <32, 32 >> >();
}

/* helper functions and data structures*/
#define checkCudaErrors(err)	__checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		exit(-1);
	}
}

namespace APUtil {
	__global__ void hookPointerAndDataKernel(SocialForceAgent** agentPtrArray, SocialForceAgent* agentArray, int numCap) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < numCap) agentPtrArray[index] = &agentArray[index];
	}
};

extern "C"
void hookPointerAndData(SocialForceAgent** agentPtrArray, SocialForceAgent* agentArray, int numCap) {
	int gSize = GRID_SIZE(numCap);
	APUtil::hookPointerAndDataKernel << <gSize, BLOCK_SIZE >> >(agentPtrArray, agentArray, numCap);
}

__device__ double SocialForceAgent::correctCrossBoader(double val, double limit)
{
	if (val >= limit)
		return limit - 0.001;
	else if (val < 0)
		return 0;
	return val;
}
void SocialForceAgent::computeIndivSocialForceRoom(const SocialForceAgentData &myData, const SocialForceAgentData &otherData, double2 &fSum){
	double cMass = 100;
	//my data
	const double2& loc = myData.loc;
	const double2& goal = myData.goal;
	const double2& velo = myData.velocity;
	const double& v0 = myData.v0;
	const double& mass = myData.mass;
	//other's data
	const double2& locOther = otherData.loc;
	const double2& goalOther = otherData.goal;
	const double2& veloOther = otherData.velocity;
	const double& v0Other = otherData.v0;
	const double& massOther = otherData.mass;

	double d = 1e-15 + sqrt((loc.x - locOther.x) * (loc.x - locOther.x) + (loc.y - locOther.y) * (loc.y - locOther.y));
	double dDelta = mass / cMass + massOther / cMass - d;
	double fExp = A * exp(dDelta / B);
	double fKg = dDelta < 0 ? 0 : k1 *dDelta;
	double nijx = (loc.x - locOther.x) / d;
	double nijy = (loc.y - locOther.y) / d;
	double fnijx = (fExp + fKg) * nijx;
	double fnijy = (fExp + fKg) * nijy;
	double fkgx = 0;
	double fkgy = 0;
	if (dDelta > 0) {
		double tix = -nijy;
		double tiy = nijx;
		fkgx = k2 * dDelta;
		fkgy = k2 * dDelta;
		double vijDelta = (veloOther.x - velo.x) * tix + (veloOther.y - velo.y) * tiy;
		fkgx = fkgx * vijDelta * tix;
		fkgy = fkgy * vijDelta * tiy;
	}
	fSum.x += fnijx + fkgx;
	fSum.y += fnijy + fkgy;
}
__device__ void SocialForceAgent::computeForceWithWall(const SocialForceAgentData &dataLocal, obstacleLine &wall, const int &cMass, double2 &fSum) {
	double2 wl = make_double2(wall.ex - wall.sx, wall.ey - wall.sy);
	if (length(wl) == 0) return;
	double diw, crx, cry;
	const double2 &loc = dataLocal.loc;

	diw = wall.pointToLineDist(loc, crx, cry);
	double virDiw = DIST(loc.x, loc.y, crx, cry);

	if (virDiw == 0)
		return;

	double niwx = (loc.x - crx) / virDiw;
	double niwy = (loc.y - cry) / virDiw;
	double drw = dataLocal.mass / cMass - diw;
	double fiw1 = A * exp(drw / B);
	if (drw > 0)
		fiw1 += k1 * drw;
	double fniwx = fiw1 * niwx;
	double fniwy = fiw1 * niwy;

	double fiwKgx = 0, fiwKgy = 0;
	if (drw > 0)
	{
		double fiwKg = k2 * drw * (dataLocal.velocity.x * (-niwy) + dataLocal.velocity.y * niwx);
		fiwKgx = fiwKg * (-niwy);
		fiwKgy = fiwKg * niwx;
	}

	fSum.x += fniwx - fiwKgx;
	fSum.y += fniwy - fiwKgy;
}
__device__ void SocialForceAgent::computeWallImpaction(const SocialForceAgentData &dataLocal, obstacleLine &wall, const double2 &newVelo, const double &tick, double &mint){
	double crx, cry, tt;
	const double2 &loc = dataLocal.loc;
	int ret = wall.intersection2LineSeg(
		loc.x,
		loc.y,
		loc.x + 0.5 * newVelo.x * tick,
		loc.y + 0.5 * newVelo.y * tick,
		crx,
		cry
		);
	if (ret == 1)
	{
		if (fabs(crx - loc.x) > 0)
			tt = (crx - loc.x) / (newVelo.x * tick);
		else
			tt = (crx - loc.y) / (newVelo.y * tick + 1e-20);
		if (tt < mint)
			mint = tt;
	}
}
__device__ void SocialForceAgent::computeDirection(const SocialForceAgentData &dataLocal, double2 &dvt) {
	//my data
	const double2& loc = dataLocal.loc;
	const double2& goal = dataLocal.goal;
	const double2& velo = dataLocal.velocity;
	const double& v0 = dataLocal.v0;
	const double& mass = dataLocal.mass;

	dvt.x = 0;	dvt.y = 0;
	double2 diff; diff.x = 0; diff.y = 0;
	double d0 = sqrt((loc.x - goal.x) * (loc.x - goal.x) + (loc.y - goal.y) * (loc.y - goal.y));
	diff.x = v0 * (goal.x - loc.x) / d0;
	diff.y = v0 * (goal.y - loc.y) / d0;
	dvt.x = (diff.x - velo.x) / tao;
	dvt.y = (diff.y - velo.y) / tao;
}

__device__ int sharedMinAndMax(int value, bool minFlag) {
	for (int i = 16; i >= 1; i /= 2) {
		if (minFlag)
			value = min(value, __shfl_xor(value, i, 32));
		else
			value = max(value, __shfl_xor(value, i, 32));
	}
	return value;
}

__device__ void SocialForceAgent::computeSocialForceRoom(SocialForceAgentData &dataLocal, double2 &fSum) {
	//__shared__ SocialForceAgentData sdata[BLOCK_SIZE];
	fSum.x = 0; fSum.y = 0;
	double ds = 0;
	int neighborCount = 0;

	for (int i = 0; i < NUM_CAP; i++) {
		SocialForceAgent *other = myClone->context[i];
		SocialForceAgentData otherData = other->data;
		ds = length(otherData.loc - dataLocal.loc);
		if (ds < 6 && ds > 0) {
			neighborCount++;
			computeIndivSocialForceRoom(dataLocal, otherData, fSum);

			for (int i = 0; i < NUM_PARAM; i++)
				this->flagCloning[i] |= other->flagCloning[i];
		}
	}

	dataLocal.numNeighbor = neighborCount;
}
__device__ void SocialForceAgent::chooseNewGoal(const double2 &newLoc, double epsilon, double2 &newGoal) {
	double2 oldGoal = newGoal;
	double2 center = make_double2(ENV_DIM / 2, ENV_DIM / 2);
	if (newLoc.x < center.x && newLoc.y <= center.y) {
		newGoal.x = 0.5 * ENV_DIM;
		newGoal.y = 0.3 * ENV_DIM;
	}
	else if (newLoc.x <= center.x && newLoc.y > center.y) {
		newGoal.x = 0.3 * ENV_DIM;
		newGoal.y = 0.5 * ENV_DIM;
	}
	else if (newLoc.x > center.x && newLoc.y > center.y) {
		newGoal.x = 0.5 * ENV_DIM;
		newGoal.y = 0.7 * ENV_DIM;
	}
	else if (newLoc.x >= center.x && newLoc.y < center.y){
		newGoal.x = 0.9 * ENV_DIM;
		newGoal.y = 0.3 * ENV_DIM;
	}
}
__device__ void SocialForceAgent::step(){

	double cMass = 100;

	const double2& loc = data.loc;
	const double2& goal = data.goal;
	const double2& velo = data.velocity;
	const double& v0 = data.v0;
	const double& mass = data.mass;

	//compute the direction
	double2 dvt;
	computeDirection(data, dvt);

	//compute force with other agents
	double2 fSum;
	computeSocialForceRoom(data, fSum);

	//compute force with walls and gates
	for (int i = 0; i < NUM_WALLS; i++) {
		obstacleLine wall = myClone->walls[i];
		computeForceWithWall(data, wall, cMass, fSum);
	}

	for (int i = 0; i < NUM_PARAM; i++) {
		obstacleLine &gate = myClone->gates[i];
		if (gate.pointToLineDist(loc) < 6) {
			// ideally, parent clone agent should compare against all child clone parameter configuration
			this->flagCloning[i] = -1;
		}
	}

	//sum up
	dvt.x += fSum.x / mass;
	dvt.y += fSum.y / mass;

	double2 newVelo = data.velocity;
	double2 newLoc = data.loc;
	double2 newGoal = data.goal;

	double tick = 0.1;
	newVelo.x += dvt.x * tick * (1);// + this->random->gaussian() * 0.1);
	newVelo.y += dvt.y * tick * (1);// + this->random->gaussian() * 0.1);
	double dv = sqrt(newVelo.x * newVelo.x + newVelo.y * newVelo.y);

	if (dv > maxv) {
		newVelo.x = newVelo.x * maxv / dv;
		newVelo.y = newVelo.y * maxv / dv;
	}

	double mint = 1;
	for (int i = 0; i < NUM_WALLS; i++) {
		obstacleLine wall = myClone->walls[i];
		computeWallImpaction(data, wall, newVelo, tick, mint);
	}

	newVelo.x *= mint;
	newVelo.y *= mint;
	newLoc.x += newVelo.x * tick;
	newLoc.y += newVelo.y * tick;

	double goalTemp = goal.x;

	chooseNewGoal(newLoc, mass / cMass, newGoal);

	newLoc.x = correctCrossBoader(newLoc.x, ENV_DIM);
	newLoc.y = correctCrossBoader(newLoc.y, ENV_DIM);

	dataCopy = data;

	dataCopy.loc = newLoc;
	dataCopy.velocity = newVelo;
	dataCopy.goal = newGoal;
}
__device__ void SocialForceAgent::init(SocialForceClone* c, int idx) {
	this->contextId = idx;
	//this->myOrigin = NULL;
	this->goalIdx = 0;
	this->myClone = c;

	for (int i = 0; i < NUM_PARAM; i++) {
		this->flagCloning[i] = 0;
		this->flagCloned[i] = 0;
	}

	curandState_t rStateLocal = c->rState[idx];
	this->color.x = curand(&rStateLocal) % 256;
	this->color.y = curand(&rStateLocal) % 256;
	this->color.z = curand(&rStateLocal) % 256;
	this->color.w = curand(&rStateLocal) % 256;
	
	SocialForceAgentData & dataLocal = this->data; //= &sfModel->originalAgents->dataArray[dataSlot];
	float rx = (float)(idx / 32) / (float)32;
	float ry = (float)(idx % 32) / (float)32;
	dataLocal.loc.x = (0.6 + 0.1 * curand_uniform(&rStateLocal)) * ENV_DIM;
	dataLocal.loc.y = (0.5 + 0.4 * curand_uniform(&rStateLocal)) * ENV_DIM;

	dataLocal.velocity.x = 2;//4 * (this->random->uniform()-0.5);
	dataLocal.velocity.y = 2;//4 * (this->random->uniform()-0.5);

	dataLocal.v0 = 2;
	dataLocal.mass = 50;
	dataLocal.numNeighbor = 0;
	//chooseNewGoal(dataLocal.loc, 0, dataLocal.goal);

	dataLocal.goal = make_double2(0.5 * ENV_DIM, 0.7 * ENV_DIM);
	this->dataCopy = dataLocal;
}

__device__ void SocialForceAgent::initNewClone(SocialForceAgent *parent, SocialForceClone *childClone) {
	this->color = childClone->color;
	this->contextId = parent->contextId;
	//this->myOrigin = parent;
	this->myClone = childClone;
	this->goalIdx = parent->goalIdx;
	for (int i = 0; i < NUM_GOAL; i++)
		this->goalSeq[i] = parent->goalSeq[i];

	for (int i = 0; i < NUM_PARAM; i++) {
		this->flagCloning[i] = 0;
		this->flagCloned[i] = 0;
	}

	this->data = parent->data;
	this->dataCopy = parent->dataCopy;

}

namespace clone {
	__global__ void stepKernel(SocialForceClone *c, int numElemLocal) {
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < numElemLocal)
			c->ap->agentPtrArray[index]->step();
	}
	__global__ void swapKernel(SocialForceClone *c, int numElemLocal) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElemLocal) {
			SocialForceAgent &agent = *c->ap->agentPtrArray[idx];
			agent.data = agent.dataCopy;
		}
	}
}

void SocialForceClone::step(int stepCount) {
	if (numElem == 0)
		return;
	int gSize;

	//alterGate(stepCount);

	/*
	cudaMemcpyAsync(contextSorted, context, sizeof(SocialForceAgent*) * NUM_CAP, cudaMemcpyDeviceToDevice, myStream);
	cudaStreamSynchronize(myStream);
	NeighborModule::sortAgentByLocKernel << <1, 1, 0, myStream >> >(this->contextSorted, this->rState, NUM_CAP);
	cudaMemsetAsync(cidStarts, 0xff, sizeof(int) * NUM_CELL * NUM_CELL, myStream);
	cudaMemsetAsync(cidEnds, 0xff, sizeof(int) * NUM_CELL * NUM_CELL, myStream);
	cudaStreamSynchronize(myStream);
	gSize = GRID_SIZE(NUM_CAP);
	NeighborModule::setCidStartEndKernel<<<gSize, BLOCK_SIZE, 0, myStream>>>(contextSorted, cidStarts, cidEnds, NUM_CAP);
	NeighborModule::sortAgentByLocKernel << <1, 1, 0, myStream >> >(this->apHost->agentPtrArray, this->rState, this->numElem);
	*/
	
	gSize = GRID_SIZE(numElem);
	size_t smemSize = sizeof(SocialForceAgentData) * BLOCK_SIZE;
	clone::stepKernel << <gSize, BLOCK_SIZE, smemSize, myStream >> >(selfDev, numElem);
	//clone::stepKernel << <gSize, BLOCK_SIZE >> >(selfDev, numElem);
}

void SocialForceClone::swap() {
	if (numElem == 0)
		return;
	int gSize = GRID_SIZE(numElem);
	clone::swapKernel << <gSize, BLOCK_SIZE >> >(selfDev, numElem);
}

void SocialForceClone::alterGate(int stepCount) {
	bool changed = false;
	for (int i = 0; i < NUM_PARAM; i++) {
		if (cloneParams[i] == stepCount) {
			changed = true;
			gates[i].init(0, 0, 0, 0);
			//cudaMemcpyAsync(&selfDev->gates[i], &gates[i], sizeof(obstacleLine), cudaMemcpyHostToDevice, myStream);
			cudaMemcpy(&selfDev->gates[i], &gates[i], sizeof(obstacleLine), cudaMemcpyHostToDevice);
		}
	}
}

namespace AppUtil {

	__device__ bool cloningCondition(SocialForceAgent *agent,
		SocialForceClone *parentClone, SocialForceClone *childClone) {

		// if agent has been cloned?
		if (childClone->cloneFlags[agent->contextId] == true)
			return false;

		// active cloning condition
		double2 &loc = agent->data.loc;
		for (int i = 0; i < NUM_PARAM; i++) {
			int param1 = parentClone->cloneParams[i];
			int param2 = childClone->cloneParams[i];
			if (param1 != param2) {
				obstacleLine g1 = parentClone->gates[i];
				obstacleLine g2 = childClone->gates[i];
				if (g1.pointToLineDist(loc) < 6)
					return true;
				if (g2.pointToLineDist(loc) < 6)
					return true;
			}
		}

		// passive cloning condition
#define MY_MAX(a, b) (a > b ? a : b)
#define MY_MIN(a, b) (a < b ? a : b)
		int minx = MY_MAX((loc.x - RADIUS_I) / CELL_DIM, 0);
		int miny = MY_MAX((loc.y - RADIUS_I) / CELL_DIM, 0);
		int maxx = MY_MIN((loc.x + RADIUS_I) / CELL_DIM, NUM_CELL - 1);
		int maxy = MY_MIN((loc.y + RADIUS_I) / CELL_DIM, NUM_CELL - 1);
		for (int i = minx; i <= maxx; i++)
			for (int j = miny; j <= maxy; j++)
				if (childClone->takenMap[i * NUM_CELL + j])
					return true;

		// pass all the check, don't need to be cloned
		return false;
	}

	__global__ void updateContextKernel(SocialForceClone *c, int numElem) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElem) {
			SocialForceAgent *agent = c->ap->agentPtrArray[idx];
			c->context[agent->contextId] = agent;
		}
	}

	__global__ void constructPassiveMap(SocialForceClone *c, int numElem) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElem) {
			SocialForceAgent &agent = *c->ap->agentPtrArray[idx];
			int takenId = agent.data.loc.x / CELL_DIM;
			takenId = takenId * NUM_CELL + agent.data.loc.y / CELL_DIM;
			c->takenMap[takenId] = true;
		}
	}

	__global__ void performCloningKernel(SocialForceClone *p, SocialForceClone *c, int numCap) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numCap) {
			SocialForceAgent *agent = p->context[idx];
			if (cloningCondition(agent, p, c)) {
				uint lastNum = atomicInc(&c->numElem, numCap);
				SocialForceAgent& childAgent = *c->ap->agentPtrArray[lastNum];
				c->ap->takenFlags[lastNum] = true;
				childAgent.initNewClone(agent, c);
				c->context[childAgent.contextId] = &childAgent;
				c->cloneFlags[childAgent.contextId] = true;
				//c->numElem++; /* not written back */
			}
		}
	}

	__global__ void performCloningOldKernel(SocialForceClone *p, SocialForceClone *c, int numParent) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numParent) {
			SocialForceAgent *agent = p->ap->agentPtrArray[idx];
			int cloneLevelLocal = c->cloneLevel;
			int cloneMaskLocal = c->cloneMasks[cloneLevelLocal];
			int cloneDecision = ~agent->flagCloned[cloneLevelLocal] & cloneMaskLocal & agent->flagCloning[cloneLevelLocal];
			if (cloneDecision > 0) {
				agent->flagCloned[cloneLevelLocal] |= cloneMaskLocal;
				agent->flagCloning[cloneLevelLocal] &= ~cloneMaskLocal;

				uint lastNum = atomicInc(&c->numElem, NUM_CAP);
				SocialForceAgent &childAgent = *c->ap->agentPtrArray[lastNum];
				c->ap->takenFlags[lastNum] = true;
				childAgent.initNewClone(agent, c);
				c->context[childAgent.contextId] = &childAgent;
				c->cloneFlags[childAgent.contextId] = true;
			}
		}
	}
	
	__global__ void compareAndEliminateKernel(SocialForceClone *p, SocialForceClone *c, int numElem) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElem) {
			SocialForceAgent &childAgent = *c->ap->agentPtrArray[idx];
			SocialForceAgent &parentAgent = *p->context[childAgent.contextId]; // *(SocialForceAgent*)childAgent.myOrigin;
			double velDiff = length(childAgent.dataCopy.velocity - parentAgent.dataCopy.velocity);
			double locDiff = length(childAgent.dataCopy.loc - parentAgent.dataCopy.loc);
			if (locDiff == 0 && velDiff	== 0) {
				c->ap->takenFlags[idx] = false;
				c->cloneFlags[childAgent.contextId] = false;
			}
		}
	}

	__global__ void compareAndEliminateOldKernel(SocialForceClone *p, SocialForceClone *c, int numElem) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < numElem) {
			SocialForceAgent &childAgent = *c->ap->agentPtrArray[idx];
			SocialForceAgent &parentAgent = *p->context[childAgent.contextId];

			double velDiff = length(childAgent.dataCopy.velocity - parentAgent.dataCopy.velocity);
			double locDiff = length(childAgent.dataCopy.loc - parentAgent.dataCopy.loc);
			if (locDiff == 0 && velDiff == 0) {
				c->ap->takenFlags[idx] = false;
				c->cloneFlags[childAgent.contextId] = false;
				int cloneLevelLocal = c->cloneLevel;
				int cloneMaskLocal = c->cloneMasks[cloneLevelLocal];

				parentAgent.flagCloned[cloneLevelLocal] &= ~cloneMaskLocal;
				parentAgent.flagCloning[cloneLevelLocal] &= ~cloneMaskLocal;
			}
		}
	}

	template<class T>
	__device__  void swap(T * ar, int a, int b) {
		T t1 = ar[a];
		ar[a] = ar[b];
		ar[b] = t1;
	}

	__global__ void reorderKernel(SocialForceClone *c, int numElem) {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx == 0) {
			int l = 0; int r = numElem;
			int i = l, j = l;
			for (; j < r; j++) {
				if (c->ap->takenFlags[j] == true) {
					swap<SocialForceAgent*>(c->ap->agentPtrArray, i, j);
					swap<bool>(c->ap->takenFlags, i, j);
					i++;
				}
			}
			c->numElem = i;
		}
	}

};

void SocialForceSimApp::performClone(SocialForceClone *parentClone, SocialForceClone *childClone) {
	childClone->parentCloneid = parentClone->cloneid;

	// 1. copy the context of parent clone
	cudaMemcpyAsync(childClone->context, parentClone->context, NUM_CAP * sizeof(SocialForceAgent*), cudaMemcpyDeviceToDevice, childClone->myStream);
	cudaStreamSynchronize(childClone->myStream);
	//cudaMemcpy(childClone->context, parentClone->context, NUM_CAP * sizeof(SocialForceAgent*), cudaMemcpyDeviceToDevice);
	getLastCudaError("perform clone");

	// 2. update the context with agents of its own
	if (childClone->numElem > 0) {
		int gSize = GRID_SIZE(childClone->numElem);
		AppUtil::updateContextKernel << <gSize, BLOCK_SIZE, 0, childClone->myStream >> >(childClone->selfDev, childClone->numElem);
		//AppUtil::updateContextKernel << <gSize, BLOCK_SIZE >> >(childClone->selfDev, childClone->numElem);
		getLastCudaError("perform clone");
	}

	// 4. perform active and passive cloning (in cloningCondition checking)
	if (parentClone->numElem > 0) {
		int gSize = GRID_SIZE(parentClone->numElem);
		//AppUtil::performCloningKernel << <gSize, BLOCK_SIZE >> >(parentClone->selfDev, childClone->selfDev, NUM_CAP);
		AppUtil::performCloningOldKernel << <gSize, BLOCK_SIZE, 0, childClone->myStream >> >(parentClone->selfDev, childClone->selfDev, parentClone->numElem);
		cudaMemcpyAsync(childClone, childClone->selfDev, sizeof(SocialForceClone), cudaMemcpyDeviceToHost, childClone->myStream);
		cudaStreamSynchronize(childClone->myStream);
		getLastCudaError("perform clone");
	}

}

void compareAndEliminateCPU(SocialForceClone *parentClone, SocialForceClone *childClone)
{
	wchar_t message[20];
	for (int i = 0; i < childClone->numElem; i++) {
		SocialForceAgent &childAgent = *childClone->ap->agentPtrArray[i];
		SocialForceAgent parentAgent; // *(SocialForceAgent*)childAgent.myOrigin;
		if (length(childAgent.dataCopy.velocity - parentAgent.dataCopy.velocity) == 0 &&
			length(childAgent.dataCopy.loc - parentAgent.dataCopy.loc) == 0) {
			childClone->ap->takenFlags[i] = false;
			childClone->cloneFlags[childAgent.contextId] = false;
		}
		/*else {
		if (childClone->cloneid == 4) {
		swprintf_s(message, 20, L"not false: %d\n", i);
		OutputDebugString(message);
		}
		}*/
	}
	childClone->numElem = childClone->ap->reorder(childClone->numElem);
}

void SocialForceSimApp::compareAndEliminate(SocialForceClone *parentClone, SocialForceClone *childClone) {
	if (childClone->numElem == 0) return;
	int gSize = GRID_SIZE(childClone->numElem);
	AppUtil::compareAndEliminateOldKernel << <gSize, BLOCK_SIZE, 0, childClone->myStream >> >(parentClone->selfDev, childClone->selfDev, childClone->numElem);
	//AppUtil::compareAndEliminateKernel << <gSize, BLOCK_SIZE>> >(parentClone->selfDev, childClone->selfDev, childClone->numElem);
	gSize = GRID_SIZE(NUM_CAP);
	AppUtil::reorderKernel << <1, 1, 0, childClone->myStream >> >(childClone->selfDev, childClone->numElem);
	//AppUtil::reorderKernel << <1, 1 >> >(childClone->selfDev, childClone->numElem);
	cudaMemcpyAsync(childClone, childClone->selfDev, sizeof(SocialForceClone), cudaMemcpyDeviceToHost, childClone->myStream);
	cudaStreamSynchronize(childClone->myStream);
}

void SocialForceSimApp::proc(int p, int c, bool o, char *s) {
	performClone(cAll[p], cAll[c]);
	cAll[c]->step(stepCount);
	if (o) {
		if (stepCount < 800)
			cAll[c]->output(stepCount, s);
	}
	compareAndEliminate(cAll[p], cAll[c]);
}

void swap(int **cloneTree, int a, int b) {
	int t1 = cloneTree[0][a];
	cloneTree[0][a] = cloneTree[0][b];
	cloneTree[0][b] = t1;

	t1 = cloneTree[1][a];
	cloneTree[1][a] = cloneTree[1][b];
	cloneTree[1][b] = t1;
}

void quickSort(int **cloneTree, int l, int r) {
	if (l == r)
		return;
	int pi = l + rand() % (r - l);
	swap(cloneTree, l, pi);
	int pivot = cloneTree[0][l];

	int i = l + 1, j = l + 1;
	for (; j < r; j++) {
		if (cloneTree[0][j] < pivot) {
			swap(cloneTree, i, j);
			i++;
		}
	}
	swap(cloneTree, l, i - 1);
	quickSort(cloneTree, l, i - 1);
	quickSort(cloneTree, i, r);
}

void SocialForceSimApp::mst() {
	// clone diff matrix
	int **cloneDiff = new int*[totalClone];
	for (int i = 0; i < totalClone; i++) {
		cloneDiff[i] = new int[totalClone];
		for (int j = 0; j < totalClone; j++)
			cloneDiff[i][j] = 0;
	}

	for (int i = 0; i < totalClone; i++) {
		for (int j = 0; j < totalClone; j++) {
			for (int k = 0; k < NUM_PARAM; k++) {
				if (cAll[i]->cloneParams[k] != cAll[j]->cloneParams[k])
					cloneDiff[i][j]++;
			}
			wchar_t message[20];
			swprintf_s(message, 20, L"%d ", cloneDiff[i][j]);
			OutputDebugString(message);
		}
		OutputDebugString(L"\n");
	}
	int *parent = cloneTree[0] = new int[totalClone];
	int *child = cloneTree[1] = new int[totalClone];
	int *key = new int[totalClone];
	bool *mstSet = new bool[totalClone];

	for (int i = 0; i < totalClone; i++)
		child[i] = i, key[i] = INT_MAX, mstSet[i] = false;

	key[0] = 0;
	parent[0] = -1;
	child[0] = 0;

	int count = 0;
	while (count++ < totalClone - 1) {
		int minKey = INT_MAX;
		int minIdx;
		for (int j = 0; j < totalClone; j++)
			if (mstSet[j] == false && key[j] < minKey)
				minKey = key[j], minIdx = j;
		mstSet[minIdx] = true;

		for (int j = 0; j < totalClone; j++)
			if (cloneDiff[minIdx][j] && mstSet[j] == false && cloneDiff[minIdx][j] < key[j])
				parent[j] = minIdx, key[j] = cloneDiff[minIdx][j];
	}

	quickSort(cloneTree, 0, totalClone);

	for (int i = 1; i < totalClone; i++) {
		wchar_t message[20];
		swprintf_s(message, 20, L"%d - %d: %d\n", cloneTree[0][i], cloneTree[1][i], cloneDiff[i][parent[i]]);
		OutputDebugString(message);
	}

	delete mstSet;
	delete key;
}

__global__ void getLocAndColorKernel(SocialForceClone *c, double2 *loc, uchar4 *color, int *contextId, int numElem) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numElem) {
		loc[idx] = c->context[idx]->data.loc;
		color[idx] = c->context[idx]->color;
		contextId[idx] = c->context[idx]->contextId;
	}
}

void SocialForceSimApp::getLocAndColorFromDevice(){
	SocialForceClone *c = cAll[paintId];
	int gSize = GRID_SIZE(NUM_CAP);
	getLocAndColorKernel << <gSize, BLOCK_SIZE >> >(c->selfDev, debugLocDev, debugColorDev, debugContextIdDev, NUM_CAP);
	cudaMemcpy(debugLocHost, debugLocDev, sizeof(double2) * NUM_CAP, cudaMemcpyDeviceToHost);
	cudaMemcpy(debugColorHost, debugColorDev, sizeof(uchar4) * NUM_CAP, cudaMemcpyDeviceToHost);
	cudaMemcpy(debugContextIdHost, debugContextIdDev, sizeof(int) * NUM_CAP, cudaMemcpyDeviceToHost);
	cudaMemcpy(c->takenMap, c->selfDev->takenMap, sizeof(bool) * NUM_CELL * NUM_CELL, cudaMemcpyDeviceToHost);
	//cudaMemcpy(debugCidStartsHost, c->cidStarts, sizeof(int) * NUM_CELL * NUM_CELL, cudaMemcpyDeviceToHost);
	//cudaMemcpy(debugCidEndsHost, c->cidEnds, sizeof(int) * NUM_CELL * NUM_CELL, cudaMemcpyDeviceToHost);
	//wchar_t message[128];
	//for (int i = 0; i < NUM_CELL * NUM_CELL; i++) {
	//	swprintf_s(message, L"(%d, %d) ", debugCidStartsHost[i], debugCidEndsHost[i]);
	//	OutputDebugString(message);
	//}
	//OutputDebugString(L"\n");
}

__global__ void initRandomKernel(SocialForceClone* c, int numElemLocal) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numElemLocal) {
		curand_init(1234, idx, 0, &c->rState[idx]);
	}
}

__global__ void initRootCloneKernel(SocialForceClone* c, int numElemLocal) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < numElemLocal) {
		c->ap->agentArray[idx].init(c, idx);
		c->context[idx] = &c->ap->agentArray[idx];
		c->cloneFlags[idx] = false;
	}
	if (idx == 0)
		c->numElem = numElemLocal;
}

void SocialForceSimApp::initRootClone(SocialForceClone* cHost, SocialForceClone* cDev) {
	cHost->numElem = NUM_CAP;

	int gSize = GRID_SIZE(NUM_CAP);
	initRandomKernel << <gSize, BLOCK_SIZE >> >(cDev, NUM_CAP);
	initRootCloneKernel << <gSize, BLOCK_SIZE >> >(cDev, NUM_CAP);
}