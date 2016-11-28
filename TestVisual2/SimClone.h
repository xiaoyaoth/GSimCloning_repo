#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

struct float2 {
	float x;
	float y;
	float2(float xx, float yy) : x(xx), y(yy) {}
	float2(){};
};

struct AgentColor{
	char r, g, b;
	AgentColor & operator = (const AgentColor & rhs) {
		r = rhs.r;
		g = rhs.g;
		b = rhs.b;
		return *this;
	}
};

class SimpleAgent {
public:
	float2 loc;
	float2 locNext;
	float2 locDefault;
	int contextId;
	AgentColor color;
};

#define NUM_CAP 1024
#define NUM_PARAM 4
#define CELL_DIM 4
#define ENV_DIM 128
#define RADIUS_I 5

class SimpleClone {
public:
	SimpleAgent *agents;
	int numElem;
	SimpleAgent **context;
	int *pv;
	AgentColor color;

	SimpleClone() {
		agents = new SimpleAgent[NUM_CAP];
		numElem = 0;
		context = new SimpleAgent*[NUM_CAP];
		pv = new int[NUM_PARAM];
		color.r = rand() % 255;
		color.g = rand() % 255;
		color.b = rand() % 255;
	}
};

inline float dot(const float2& a, const float2& b)
{
	return a.x * b.x + a.y * b.y;
}

inline float length(const float2& v)
{
	return sqrtf(dot(v, v));
}

inline float2 operator-(const float2& a, const float2& b)
{
	return float2(a.x - b.x, a.y - b.y);
}

//inline int max(int a, int b)
//{
//	return a > b ? a : b;
//}
//
//inline int min(int a, int b)
//{
//	return a < b ? a : b;
//}


class SimCloneApp {
public:
	SimpleClone **cAll;
	SimpleAgent *agents;
	SimpleAgent **context;
	int paintId;

	void step(SimpleClone *clone) {
		float2 c1(32, 32), c2(32, 96), c3(96, 32), c4(96, 96);
		for (int i = 0; i < clone->numElem; i++) {
			SimpleAgent &agent = clone->agents[i];
			if (clone->pv[0] == 1 && length(agent.loc - c1) < 4) {
				if (agent.loc.y < c1.y)
					agent.locNext.y = agent.loc.y - 0.2;
				else
					agent.locNext.y = agent.loc.y + 0.2;
				continue;
			}
			if (clone->pv[1] == 1 && length(agent.loc - c2) < 4) {
				if (agent.loc.y < c2.y)
					agent.locNext.y = agent.loc.y - 0.2;
				else
					agent.locNext.y = agent.loc.y + 0.2;
				continue;
			}
			if (clone->pv[2] == 1 && length(agent.loc - c3) < 4) {
				if (agent.loc.y < c3.y)
					agent.locNext.y = agent.loc.y - 0.2;
				else
					agent.locNext.y = agent.loc.y + 0.2;
				continue;
			}
			if (clone->pv[3] == 1 && length(agent.loc - c4) < 4) {
				if (agent.loc.y < c4.y)
					agent.locNext.y = agent.loc.y - 0.2;
				else
					agent.locNext.y = agent.loc.y + 0.2;
				continue;
			}
			if (agent.loc.y > agent.locDefault.y)
				agent.locNext.y = agent.loc.y - 0.2;
			else if (agent.loc.y < agent.locDefault.y)
				agent.locNext.y = agent.loc.y + 0.2;
			agent.locNext.x = agent.loc.x + 0.2;
		}
	}

	void swapAll(SimpleClone *clone) {
		for (int i = 0; i < clone->numElem; i++) {
			SimpleAgent &agent = clone->agents[i];
			agent.loc = agent.locNext;
		}
	}

	bool cloningCondition(SimpleAgent *agent, bool *childTakenMap,
		SimpleClone *parentClone, SimpleClone *childClone) {
		float2 c1(32, 32), c2(32, 96), c3(96, 32), c4(96, 96);
		// if agent has been cloned?
		if (agent != childClone->context[agent->contextId])
			return false;

		// active cloning condition
		float2 &loc = agent->loc;
		if (parentClone->pv[0] != childClone->pv[0])
			if (length(loc - c1) < 6) return true;
		if (parentClone->pv[1] != childClone->pv[1])
			if (length(loc - c2) < 8) return true;
		if (parentClone->pv[2] != childClone->pv[2])
			if (length(loc - c3) < 10) return true;
		if (parentClone->pv[3] != childClone->pv[3])
			if (length(loc - c4) < 12) return true;

		// passive cloning condition
		/*
		int minx = max((loc.x - RADIUS_I) / (ENV_DIM / CELL_DIM), 0);
		int miny = max((loc.y - RADIUS_I) / (ENV_DIM / CELL_DIM), 0);
		int maxx = min((loc.x + RADIUS_I) / (ENV_DIM / CELL_DIM), CELL_DIM - 1);
		int maxy = min((loc.y + RADIUS_I) / (ENV_DIM / CELL_DIM), CELL_DIM - 1);
		for (int i = minx; i <= maxx; i++)
		for (int j = miny; j <= maxy; j++)
		if (childTakenMap[i * CELL_DIM + j])
		return true;
		*/

		// pass all the check, don't need to be cloned
		return false;
	}

	void performClone(SimpleClone *parentClone, SimpleClone *childClone) {
		// 1. copy the context of parent clone
		memcpy(childClone->context, parentClone->context, NUM_CAP * sizeof(void*));

		// 2. update the context with agents of its own
		for (int i = 0; i < childClone->numElem; i++) {
			SimpleAgent *agent = &childClone->agents[i];
			childClone->context[agent->contextId] = agent;
		}

		// 3. Perform cloning
		// 3.1 construct passive cloning map
		float2 dim(ENV_DIM, ENV_DIM);
		bool *takenMap = new bool[CELL_DIM * CELL_DIM];
		memset(takenMap, 0, sizeof(bool) * CELL_DIM * CELL_DIM);
		for (int i = 0; i < childClone->numElem; i++) {
			const SimpleAgent &agent = childClone->agents[i];
			int takenId = agent.loc.x / ENV_DIM;
			takenId = takenId * CELL_DIM + agent.loc.y / ENV_DIM;
			takenMap[takenId] = true;
		}

		// 3.2 perform cloning
		for (int i = 0; i < NUM_CAP; i++) {
			SimpleAgent *agent = parentClone->context[i];
			if (cloningCondition(agent, takenMap, parentClone, childClone)) {
				SimpleAgent &childAgent = childClone->agents[childClone->numElem];
				childAgent = *agent;
				childAgent.color = childClone->color;
				childClone->context[childAgent.contextId] = &childAgent;
				childClone->numElem++;
			}
		}
		delete takenMap;
	}

	int initSimClone() {
		srand(time(NULL));
		//// init root clone;
		//SimpleClone *rootClone = new SimpleClone();
		//rootClone->pv[0] = 0;
		//rootClone->pv[1] = 0;
		//rootClone->pv[2] = 0;
		//rootClone->pv[3] = 0;

		agents = new SimpleAgent[NUM_CAP];
		context = new SimpleAgent*[NUM_CAP];
		paintId = 0;

		cAll = new SimpleClone*[16];
		for (int i = 0; i < 16; i++) {
			cAll[i] = new SimpleClone();
			cAll[i]->pv[0] = i & 1;
			cAll[i]->pv[1] = (i >> 1) & 1;
			cAll[i]->pv[2] = (i >> 2) & 1;
			cAll[i]->pv[3] = (i >> 3) & 1;
			cout << cAll[i]->pv[0]
				<< cAll[i]->pv[1]
				<< cAll[i]->pv[2]
				<< cAll[i]->pv[3] << " ";
		}
		cout << endl;

		for (int i = 0; i < NUM_CAP; i++) {
			//float rx = (float)rand() / (float)RAND_MAX;
			//float ry = (float)rand() / (float)RAND_MAX;
			float rx = (float)(i / 32) / (float)32;
			float ry = (float)(i % 32) / (float)32;
			agents[i].loc = float2(rx * ENV_DIM, ry * ENV_DIM);
			agents[i].locNext = float2(rx * ENV_DIM, ry * ENV_DIM);
			agents[i].locDefault = float2(rx * ENV_DIM, ry * ENV_DIM);
			agents[i].contextId = i;
			agents[i].color = cAll[0]->color;
			context[i] = &agents[i];
		}

		return EXIT_SUCCESS;
	}

	void stepApp() {
		//cAll[15]->context = context;
		//performClone(cAll[15], cAll[10]);
		//cout << cAll[10]->numElem << " ";
		//for (int i = 0; i < 16; i++) {

		//int stepCount = 0;

		int i = 0;
		//for (int i = 0; i < 16; i++) {
		cAll[i]->context = context;
		cAll[i]->agents = agents;
		cAll[i]->numElem = NUM_CAP;

		//while (stepCount++ < 100) {
		step(cAll[i]);
		for (int j = 0; j < 16; j++) {
			performClone(cAll[i], cAll[j]);
			step(cAll[j]);
			cout << cAll[j]->numElem << " ";
		}
		for (int j = 0; j < 16; j++) {
			swapAll(cAll[j]);
		}
		cout << endl;
		//}
		//}

		/* start processing */
		//int stepCount = 0;
		//cAll[0]->context = context;
		//cAll[0]->agents = agents;
		//cAll[0]->numElem = NUM_CAP;
		//while (stepCount++ < 100) {
		//	step(cAll[0]);
		//	performClone(cAll[0], cAll[1]);
		//	step(cAll[1]);
		//	performClone(cAll[1], cAll[2]);
		//	step(cAll[2]);

		//	cout << cAll[2]->numElem << " " << cAll[2]->agents[0].loc.x << " "
		//		<< cAll[2]->agents[0].loc.y << endl;
		//	
		//	swapAll(cAll[0]);
		//	swapAll(cAll[1]);
		//	swapAll(cAll[2]);
		//}
	}
};
