#include "../TestVisual2/SocialForceGPU.h"
#include <iostream>

//kernel_4_exp3_concurrency is used for experiment

using namespace std;

double PCFreq = 0.0;
__int64 CounterStart = 0;

void StartCounter()
{
	LARGE_INTEGER li;
	if (!QueryPerformanceFrequency(&li))
		cout << "QueryPerformanceFrequency failed!\n";

	PCFreq = double(li.QuadPart) / 1000.0;

	QueryPerformanceCounter(&li);
	CounterStart = li.QuadPart;
}
double GetCounter()
{
	LARGE_INTEGER li;
	QueryPerformanceCounter(&li);
	return double(li.QuadPart - CounterStart) / PCFreq;
}

int main(int argc, char **argv) {
	fstream fout;
	fout.open("exp3.txt", fstream::out);

	SocialForceSimApp app1;
	app1.initSimClone();
	StartCounter();
	for (int i = 0; i < 1000; i++) {
		cout << i << " ";
		app1.stepApp();
		//fout << i / 4 << "\t";
		fout << GetCounter() << "\t";
		int na = 0;
		for (int i = 0; i < app1.totalClone; i++) {
			na += app1.cAll[i]->numElem;
		}
		fout << na << endl;
	}
	cout << "step1: " << (GetCounter()) << endl;
	fout.close();
	return 0;
}
