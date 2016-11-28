#ifndef SIMPLE_CLONE_CUH
#define SIMPLE_CLONE_CUH

#include "gsimcore.cuh"
#include "gsimvisual.cuh"

#define NUM_AGENT 256
#define BLOCK_SIZE 64
#define GRID_SIZE(n) (n%BLOCK_SIZE==0 ? n/BLOCK_SIZE : n/BLOCK_SIZE + 1)
#define NUM_PARAM 4

class SimpleAgent;
class SimpleModel;
class SimpleClone;
__global__ void addRootCloneAgents(SimpleClone *rootCloneDev);
__global__ void activeCloneKernel(SimpleClone *parentCloneDev, SimpleClone *childCloneDev);
__global__ void updateContextKernel(SimpleClone *childCloneDev);

struct SimpleAgentData : public GAgentData {

};

struct obstacle {

};

class SimpleClone {
public:
	obstacle *obst, *obstHost;
	AgentPool<SimpleAgent, SimpleAgentData> *agents, *agentsHost;
	SimpleAgent **cloneContext;

	SimpleClone *selfDev;
	SimpleClone *parentClone;
	thrust::host_vector<SimpleClone*> *childClones;

	uchar4 cloneColor;

	SimpleModel *myModel, *myModelHost;
	int *pv, *pvHost; // parameter value

	__host__ SimpleClone(SimpleModel *modelHost, SimpleClone *parent, int *pvInput) {
		myModelHost = modelHost;
		myModel = (SimpleModel*)modelHost->model;

		pvHost = new int[NUM_PARAM];
		cudaMalloc((void**)&pvHost, sizeof(int) * NUM_PARAM);
		memcpy(pvHost, pvInput, sizeof(int) * NUM_PARAM);
		cudaMemcpy(pv, pvInput, sizeof(int) * NUM_PARAM, cudaMemcpyHostToDevice);

		parentClone = parent;
		childClones = new thrust::host_vector<SimpleClone*>();

		agentsHost = new AgentPool<SimpleAgent, SimpleAgentData>(0, modelHostParams.MAX_AGENT_NO, sizeof(SimpleAgentData));
		util::hostAllocCopyToDevice<AgentPool<SimpleAgent, SimpleAgentData> >(agentsHost, &agents);

		int r = rand();
		memcpy(&cloneColor, &r, sizeof(uchar4));

		util::hostAllocCopyToDevice<SimpleClone>(this, &selfDev);
	}

	__host__ void step();
	__host__ void activeClone();
	__host__ void updateContext();
	__host__ void throttling();
	__device__ bool cloningConditionDev(SimpleAgent *parentAgent);
	__device__ void updateContextDev(SimpleAgent *childAgent);
};

class SimpleModel : public GModel {
public:
	SimpleClone *rootClone;
	GRandom *random, *randomHost;
	thrust::host_vector<SimpleClone*> *clones;

	__host__ SimpleModel() {
		rootClone = new SimpleClone(this, NULL, NULL);
		clones = new thrust::host_vector<SimpleClone*>();

		randomHost = new GRandom(modelHostParams.MAX_AGENT_NO);
		util::hostAllocCopyToDevice<GRandom>(randomHost, &random);

		util::hostAllocCopyToDevice<SimpleModel>(this, (SimpleModel**)&model);

		int gSize = GRID_SIZE(NUM_AGENT);
		addRootCloneAgents << <gSize, BLOCK_SIZE >> > (rootClone->selfDev);
	}

	__host__ void start() {}

	__host__ void preStep() {}

	__host__ void step() {
		// all clones are launched, but child clones need to wait for the synchronization barrier
		for (int i = 0; i < (*clones).size(); i++)
			(*clones)[i]->step();

		if (stepCount == 10) { // if certain criterion is met, for example clone 0 is chosen
			SimpleClone *c = new SimpleClone(this, NULL, NULL);
			c->parentClone = rootClone;
			rootClone->childClones->push_back(c);
			(*clones).push_back(c);
		}
		//rootClone->step(this);
	}

	__host__ void stop() {}
};

class SimpleAgent : public GAgent {
public:
	SimpleAgentData *data;
	SimpleAgentData *dataCopy;

	int ptrInPool;
	uchar4 color;

	SimpleClone *myClone;
	SimpleModel *myModel;

	int contextId;
	SimpleAgent *myOrigin;

public:
	__device__ void swapDataAndCopy(){
		SimpleAgentData *temp = this->data;
		this->data = this->dataCopy;
		this->dataCopy = temp;
	}
	__device__ int locHash() {
		float2 myLoc = this->data->loc;
		int xhash = (int)(myLoc.x / modelDevParams.CLEN_X);
		int yhash = (int)(myLoc.y / modelDevParams.CLEN_Y);
		return util::zcode(xhash, yhash);
	}

	__device__ void step(GModel *model) {
		//doSomethingWith();
		SimpleModel *myModel = (SimpleModel*)model;
		myClone->obst;
		myModel->random->uniform();
	}

	__device__ void init(int dataSlot, SimpleClone *myClone) {
		this->color = myClone->cloneColor;
		this->contextId = dataSlot;

		this->myOrigin = NULL;
		this->myClone = myClone;

		SimpleAgentData dataLocal; //= &sfModel->originalAgents->dataArray[dataSlot];
		dataLocal.agentPtr = this;

		this->data = myClone->agents->dataInSlot(dataSlot);
		this->dataCopy = myClone->agents->dataCopyInSlot(dataSlot);
		*(SimpleAgentData*)this->data = dataLocal;
		*(SimpleAgentData*)this->dataCopy = dataLocal;
	}

	__device__ void initWithParent(SimpleAgent *parentAgent, int dataSlot, SimpleClone *myClone) {
		this->color = myClone->cloneColor;
		this->contextId = parentAgent->contextId;
		this->myOrigin = parentAgent;
		this->myClone = myClone;
		SimpleAgentData dataLocal = *(SimpleAgentData*)parentAgent->data;
		dataLocal.agentPtr = this;

		this->data = myClone->agents->dataInSlot(dataSlot);
		this->dataCopy = myClone->agents->dataCopyInSlot(dataSlot);
		*(SimpleAgentData*)this->data = dataLocal;
		*(SimpleAgentData*)this->dataCopy = dataLocal;

	}
};

__device__ bool cloningConditionDev(SimpleClone *parentClone, SimpleClone *childClone,
	SimpleAgent *parentAgent, SimpleAgent *childAgent) {
	int i = 0;
	float2 parentLoc = parentAgent->data->loc;
	float2 dim = make_float2(modelDevParams.WIDTH, modelDevParams.HEIGHT);
	if (parentClone->pv[i] != childClone->pv[i]) {
		if (length(parentLoc - 0.25 * dim) < 10)
			return true;
	}
	i++; // i == 1
	if (parentClone->pv[i] != childClone->pv[i]) {
		if (length(parentLoc - make_float2(0.25 * dim.x + 0.75 * dim.y)) < 10)
			return true;
	}
	i++; // i == 2
	if (parentClone->pv[i] != childClone->pv[i]) {
		if (length(parentLoc - make_float2(0.25 * dim.x + 0.75 * dim.y)) < 10)
			return true;
	}
	i++; // i == 3
	if (parentClone->pv[i] != childClone->pv[i]) {
		if (length(parentLoc - 0.75 * dim) < 10)
			return true;
	}
}

__device__ bool SimpleClone::cloningConditionDev(SimpleAgent *parentAgent){
	// check if a given parent need to be cloned
	// 1. check new parameter with parent agent
	// doSomethingWith(this->obst, parentAgent);
	// 2. check existing child agents with parent agent
	// doSomethingWith(this->agents, parentAgent);
	float r = this->myModel->random->uniform();
	return r < 0.2;
}

__host__ void SimpleClone::activeClone() {
	int gSize = GRID_SIZE(NUM_AGENT); // NUM_AGENT is the number of agent in the context
	activeCloneKernel << <gSize, BLOCK_SIZE >> >(parentClone->selfDev, this->selfDev);
}

__host__ void SimpleClone::updateContext() {
	cudaMemcpy(this->cloneContext, parentClone->cloneContext,
		sizeof(SimpleAgent*) * NUM_AGENT, cudaMemcpyDeviceToDevice);
	int gSize = GRID_SIZE(NUM_AGENT); // NUM_AGENT is the number of agent in the context
	updateContextKernel << <gSize, BLOCK_SIZE >> > (this->selfDev);
}

__host__ void SimpleClone::throttling() {
	// perform throttling mechanism to refrain the cloning propagation
}

__host__ void SimpleClone::step() {
	if (this->parentClone != NULL) {
		this->activeClone();
	}
	this->agentsHost->cleanup(this->agents);
	if (this->agentsHost->numElem > 0) {
		this->updateContext();
		this->agentsHost->stepPoolAgent(myModelHost->model, 0);
		this->throttling();
	}
}

__global__ void addRootCloneAgents(SimpleClone *rootCloneDev)
{
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < NUM_AGENT){
		int dataSlot = idx;
		SimpleAgent *ag = rootCloneDev->agents->agentInSlot(dataSlot);
		ag->init(dataSlot, rootCloneDev);
		rootCloneDev->agents->add(ag, idx);
	}
}

__global__ void activeCloneKernel(SimpleClone *parentCloneDev, SimpleClone *childCloneDev) {
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < NUM_AGENT) { // number of agent in the context
		SimpleAgent *parentAgent = parentCloneDev->agents->agentPtrArray[idx];
		// doSomethingWith parentAgent and the parameter
		bool cloningFlag = childCloneDev->cloningConditionDev(parentAgent);
		if (cloningFlag) { // active clone condition
			int agentSlot = childCloneDev->agents->agentSlot();
			int dataSlot = childCloneDev->agents->dataSlot(agentSlot);
			SimpleAgent *childAgent = childCloneDev->agents->agentInSlot(dataSlot);
			childAgent->initWithParent(parentAgent, dataSlot, childCloneDev);
			childCloneDev->agents->add(childAgent, agentSlot);
		}
	}
}

__global__ void updateContextKernel(SimpleClone *childCloneDev) {
	uint idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < childCloneDev->agents->numElem) {
		SimpleAgent *childAgent = childCloneDev->agents->agentPtrArray[idx];
		childCloneDev->updateContextDev(childAgent);
		childCloneDev->cloneContext[childAgent->contextId] = childAgent;
	}
}

#endif