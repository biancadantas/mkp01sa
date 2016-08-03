/*
 * Descrição: Código em CUDA para solução do problema da mochila
 * 			  multidimensional utilizando Simulated Annealing.
 * 		      Baseado nos artigos:
 * 			  - Simulated Annealing for the 0/1 Multidimensional
 *              Knapsack Problem (Qian Fubin, Ding Rui)
 *            - An Efficient Implementation of Parallel Simulated
 *              Annealing Algorithm in GPUs (A.M Ferreiro, J.A.
 *              García, J.G. López-Salas, C. Vázquez) - J. Glob.
 *              Optim. (2013) 57:863-890
 * Versão: Assíncrona
 * Programadores: Bianca de Almeida Dantas
 * Criação: 03/2015
 * Última modificação: 16/04/2016
 * Alterações: adição das memórias de textura e constante.
 * Versão: adiciona só o primeiro encontrado após a retirada do escolhido
 */

/*
 * Inclusão de bibliotecas
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <sys/time.h>
#include <time.h>
#include <math.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#define EXECS 30

////////////////////////////////////
//Constantes
////////////////////////////////////
__constant__ int c_numItems[1];
__constant__ int c_numResources[1];
__constant__ int c_optSol[1];

////////////////////////////////////
//Texturas
////////////////////////////////////
texture<int, 1, cudaReadModeElementType> tex_values;
texture<int, 1, cudaReadModeElementType> tex_useResources;
texture<int, 1, cudaReadModeElementType> tex_resources;

static void HandleError(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line );
		//system("pause");
        exit(EXIT_FAILURE);
    }
}

#define CHECK_ERROR(err) (HandleError(err, __FILE__, __LINE__))

 /*
 * Definição da Mochila Multidimensional
 */
struct Mochila
{
    int numItems;        // Número de itens
    int numResources;  	 // Número de recursos
    int* values;       // Valores dos itens
    int* useResources;  // O quanto cada item consome de cada recurso (matriz em forma de vetor)
    int* resources;      // Quantidade disponível de cada recurso
};

/*
 * Declaração de variáveis globais
 */
Mochila* h_m;

int* dev_m_values;
int* dev_m_useResources;
int* dev_m_resources;

int numberOfIterations;
int bestIteration;
int bestSolValue;
int optSol;
float error;

double* vetGaps;
double* vetTimes;
double mediumGap;
double mediumTime;
double dpTimes;
double dpGaps;
int bestOverallSolValue;
double bestOverallTime;
double bestOverallGap;
int numOpt;
int numBetter;

int* h_sol;
int* h_incSol;
int* h_newSol;
//int* h_bestSol;
float* h_psUt;
int* h_rc;
int* h_incRc;
int* h_newRc;

int* d_sol;
int* d_incSol;
int* d_newSol;
//int* d_bestSol;
float* d_psUt;
int* d_rc;
int* d_incRc;
int* d_newRc;
int* d_rclIndexes;

int* d_tested;

//curandState st;
curandState* devStates;

float alpha; //Parâmetro que determina os elementos presentes na RCL
float delta;
float factor;
double temp;
double initTemp;
double finalTemp;
int chainLength;
int maxChainLength;

int* h_solValue;
int* h_incSolValue;
//int* h_newSolValue;
int* h_bestSolValue;
int* d_solValue;
int* d_incSolValue;
//int* d_newSolValue;
int* d_bestSolValue;

/* Declaração de variáveis no device */
//Mochila* d_m;

int numBlocks;
int numThreads;

/*
 * Prototipação das funções
 */
__global__ void setupKernel(curandState*, unsigned long);
__global__ void replicateSolution(int*, int*, int*, int);
__global__ void markovChainKernel(int*, int*, int*, int*, int*, curandState*, int, int*, double, double, float);
__device__ bool allTested(int*);
__global__ void initialize(int*);
__device__ void calculatePseudoUtilities(int*, float*, int*);
__device__ bool canAdd(int, int*, int*);
__device__ int getSolutionValue(int*);
__global__ void constructSolution(int*, int*, int*, float*, float, curandState*, int*);
__device__ bool isFeasibleAdd(int*, int);
__device__ bool isFeasibleRem(int*);
__device__ int getMaxPsUtIndex(float*);
__device__ int getMinPsUtIndex(float*);
__host__ void writeBestResult(FILE*, char*, double, int, int);
__host__ void finalWrite(FILE*, char*, double, double, double, double, double, int, int);
__host__ int findBestSolIndex(int*);

 /*
  * Função principal
  */
int main(int argc, char *argv[])
{
	//Verifica se o número de argumentos do programa está correto
    if (argc != 9)
    {
		printf("Uso:\n./cudasa arq_teste t0 tf markov_length temp_tax alpha_rcl numBlocks numThreads\n");

		return 0;
    }

	//Seta a taxa de aprendizado e o número máximo de épocas
	initTemp = atof(argv[2]);
	finalTemp = atof(argv[3]);
	maxChainLength = atoi(argv[4]);
	factor = atof(argv[5]);
	alpha = atof(argv[6]);

    FILE* fileIn = NULL;
    FILE* fileOut = NULL;

    int j, k, exec, maxIndex;
    //float maxValue;
    //timeval start, finish;
	float elapsedTimeGPU;
	cudaEvent_t e_Start, e_Stop;

    double totalTime; // Variaveis auxiliares para a tomada de tempo
    char nomeArq[20];

    strcpy(nomeArq, argv[1]);

    //seed = 1;

    //totalError = 0.0;

	char fileInterName[50];
    sprintf(fileInterName, "%s%s", nomeArq, "int");

	//Especifica as dimensões do ambiente CUDA
	numBlocks = atoi(argv[7]);
	numThreads = atoi(argv[8]);

    char fileOutName[55];
    sprintf(fileOutName, "cudasa%s_blk%d_thr%d.out", nomeArq, numBlocks, numThreads);

	//Abertura do arquivo de entrada
	fileIn = fopen(nomeArq, "r");

	if (fileIn == NULL)
	{
	    printf("Erro na abertura do arquivo de entrada.\n");
	    return 1;
	}

	/*******************************************************************
	 * Início da leitura de dados
	 ******************************************************************/
	h_m = (Mochila*) malloc(sizeof(Mochila));

	//Leitura dos dados de entrada
	//int numItems, numResources;
	fscanf(fileIn, "%d", &h_m->numItems);
	fscanf(fileIn, "%d", &h_m->numResources);

	//printf("%d ", h_m->numItems);
	////system("pause");

	fscanf(fileIn, "%d", &(optSol));

	//Alocação do vetor de values
	h_m->values = (int*) malloc(h_m->numItems * sizeof(int));

	//Leitura do vetor de values
	for (j = 0; j < h_m->numItems; j++)
		fscanf(fileIn, "%d", &(h_m->values[j]));

	//Alocação da matriz de useResources como um vetor
	h_m->useResources = (int*) malloc(h_m->numItems * h_m->numResources * sizeof(int));

	//Leitura da matriz de useResources
	//Agora está sendo armazenado em N linhas e M colunas
	for (j = 0; j < h_m->numResources; j++)
	{
		//displ = j * h_m->numItems;
		for (k = 0; k < h_m->numItems; k++)
			fscanf(fileIn, "%d", &(h_m->useResources[k * h_m->numResources + j]));
			//fscanf(fileIn, "%d", &(h_m->useResources[displ + k]));
	}

	//Alocação do vetor de recursos
	h_m->resources = (int*) malloc(h_m->numResources * sizeof(int));

	//Leitura do vetor de recursos
	for (j = 0; j < h_m->numResources; j++)
		fscanf(fileIn, "%d", &(h_m->resources[j]));

	//Especifica as dimensões do ambiente CUDA
	dim3 tamGrid(numBlocks, 1, 1);
	dim3 tamBlock(numThreads, 1, 1);

	CHECK_ERROR(cudaMalloc((void**) &devStates, numBlocks * numThreads * sizeof(curandState)));

	//Criando a mochila no device
	CHECK_ERROR(cudaMalloc((void**) &dev_m_values, h_m->numItems * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &dev_m_useResources, h_m->numItems * h_m->numResources * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &dev_m_resources, h_m->numResources * sizeof(int)));

	//Copiando dados da mochila para o device
	CHECK_ERROR(cudaMemcpy(dev_m_values, h_m->values, h_m->numItems * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(dev_m_useResources, h_m->useResources, h_m->numItems * h_m->numResources * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_ERROR(cudaMemcpy(dev_m_resources, h_m->resources, h_m->numResources * sizeof(int), cudaMemcpyHostToDevice));

	//Alocação do vetor solução
	h_sol = (int*) malloc(numBlocks * numThreads * h_m->numItems * sizeof(int));
	//h_bestSol = (int*) malloc(numBlocks * numThreads * h_m->numItems * sizeof(int));
	//h_incSol = (int*) malloc(numBlocks * h_m->numItems * sizeof(int));
	h_newSol = (int*) malloc(numBlocks * numThreads * h_m->numItems * sizeof(int));
	//h_incRc = (int*) malloc(numBlocks * h_m->numResources * sizeof(int));
	h_newRc = (int*) malloc(numBlocks * numThreads * h_m->numResources * sizeof(int));

	//Alocação do vetor de pseudoutilidades e do vetor de estruturas
	//auxiliar para ordenação
	h_psUt = (float*) malloc(numBlocks * numThreads * h_m->numItems * sizeof(float));

	h_solValue = (int*) malloc(numBlocks * numThreads * sizeof(int));
	//h_incSolValue = (int*) malloc(numBlocks * sizeof(int));

	//Alocação do vetor que armazena o quanto de cada recurso ainda está
	//disponível
	h_rc = (int*) malloc(numBlocks * numThreads * h_m->numResources * sizeof(int));

	CHECK_ERROR(cudaMalloc((void**) &d_sol, numBlocks * numThreads * h_m->numItems * sizeof(int)));
	//CHECK_ERROR(cudaMalloc((void**) &d_bestSol, numBlocks * h_m->numItems * sizeof(int)));
	//CHECK_ERROR(cudaMalloc((void**) &d_incSol, numBlocks * h_m->numItems * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &d_newSol, numBlocks * numThreads * h_m->numItems * sizeof(int)));

	CHECK_ERROR(cudaMalloc((void**) &d_rc, numBlocks * numThreads * h_m->numResources * sizeof(int)));
	//CHECK_ERROR(cudaMalloc((void**) &d_incRc, numBlocks * h_m->numResources * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &d_newRc, numBlocks * numThreads * h_m->numResources * sizeof(int)));

	CHECK_ERROR(cudaMalloc((void**) &d_psUt, numBlocks * numThreads * h_m->numItems * sizeof(float)));
	CHECK_ERROR(cudaMalloc((void**) &d_solValue, numBlocks * numThreads * sizeof(int)));
	//CHECK_ERROR(cudaMalloc((void**) &d_incSolValue, numBlocks * sizeof(int)));
	//CHECK_ERROR(cudaMalloc((void**) &d_newSolValue, numBlocks * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &d_rclIndexes, numBlocks * numThreads * h_m->numItems * sizeof(int)));
	CHECK_ERROR(cudaMalloc((void**) &d_tested, numBlocks * numThreads * h_m->numItems * sizeof(int)));

	//fflush(stdin);
	//printf("Leitura e alocação encerradas.\n");

	//Liga as texturas aos dados da mochila
	cudaBindTexture(0, tex_values, dev_m_values, h_m->numItems * sizeof(int));
	cudaBindTexture(0, tex_useResources, dev_m_useResources, h_m->numItems * h_m->numResources * sizeof(int));
	cudaBindTexture(0, tex_resources, dev_m_resources, h_m->numResources * sizeof(int));

	//Seta a memória constante
	cudaMemcpyToSymbol(c_numItems, &h_m->numItems, 1 * sizeof(int));
	cudaMemcpyToSymbol(c_numResources, &h_m->numResources, 1 * sizeof(int));	
	cudaMemcpyToSymbol(c_optSol, &optSol, 1 * sizeof(int));	

	vetGaps = (double*) malloc(EXECS * sizeof(double));
	vetTimes = (double*) malloc(EXECS * sizeof(double));
	mediumTime = 0.0;
	mediumGap = 0.0;
	bestOverallSolValue = 0;
	numOpt = 0;
	numBetter = 0;

	/*******************************************************************
	 * Laço de execuções                                               *
	 ******************************************************************/
	for (exec = 0; exec < EXECS; exec++)
	{
		bestSolValue = 0;

		//Inicia contagem do tempo de processamento na GPU
		elapsedTimeGPU = 0.0;

		cudaEventCreate(&e_Start);
		cudaEventCreate(&e_Stop);
		cudaEventRecord(e_Start, cudaEventDefault);

		setupKernel<<<tamGrid, tamBlock>>>(devStates, time(NULL));
		//CHECK_ERROR(cudaDeviceSynchronize());

		/*******************************************
		 * Obtenção da solução inicial.
		 * Utilizou-se a construção da solução
		 * inicial do GRASP
		 *******************************************/
		constructSolution<<<tamGrid, tamBlock>>>(d_sol, d_solValue, d_rc, d_psUt, alpha, devStates, d_rclIndexes);
		//CHECK_ERROR(cudaDeviceSynchronize());
		CHECK_ERROR(cudaMemcpy(d_newSol, d_sol, numBlocks * numThreads * h_m->numItems * sizeof(int), cudaMemcpyDeviceToDevice));
		CHECK_ERROR(cudaMemcpy(d_newRc, d_rc, numBlocks * numThreads * h_m->numResources * sizeof(int), cudaMemcpyDeviceToDevice));

		//for (int i = 0; i < numBlocks * numThreads; i++)
		//	printf("%d ", h_solValue[i]);
		//printf("\n");
		//system("pause");

		//maxIndex = findBestSolIndex(h_solValue);

		//Replica a melhor solução para todas as threads
		//replicateSolution<<<tamGrid, tamBlock>>>(d_sol, d_rc, d_solValue, maxIndex);
		//CHECK_ERROR(cudaDeviceSynchronize());

		/*******************************************
		 * Laço principal do SA.
		 * Executado até alcançar a temperatura final
		 *******************************************/
		//printf("T0: %.2f Tf: %.6f M: %d Taxa: %.2f Alpha: %.2f\n",
		//		initTemp, finalTemp, maxChainLength, factor, alpha);
		//temp = initTemp;
		int num = maxChainLength / (numBlocks * numThreads);

		//printf("%d\n", num);

		//while (temp >= finalTemp)
		//{
			//printf("\nMarkov (t = %f)", temp);
			//CHECK_ERROR(cudaMemcpy(d_newSol, d_sol, numBlocks * numThreads * h_m->numItems * sizeof(int), cudaMemcpyDeviceToDevice));
			//CHECK_ERROR(cudaMemcpy(d_newRc, d_rc, numBlocks * numThreads * h_m->numResources * sizeof(int), cudaMemcpyDeviceToDevice));
			//Kernel que computa as cadeias de Markov

			markovChainKernel<<<tamGrid, tamBlock>>>(d_sol, d_newSol, d_rc, d_newRc, d_solValue, devStates, 
				                                     num, d_tested, initTemp, finalTemp, factor);
			//CHECK_ERROR(cudaDeviceSynchronize());
			//CHECK_ERROR(cudaMemcpy(h_solValue, d_solValue, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost));
			//CHECK_ERROR(cudaMemcpy(h_sol, d_sol, numBlocks * numThreads * h_m->numItems * sizeof(int), cudaMemcpyDeviceToHost));

			//for (int i = 0; i < numBlocks * numThreads; i++)
				//printf("%d ", h_solValue[i]);
			//printf("\n");
			//system("pause");

			//maxIndex = findBestSolIndex(h_solValue);
			//bestSolThreadValue = h_bestSolValues[bestSolThreadIndex];

			//printf("\nMaior indice: %d Maior valor: %d", maxIndex, h_solValue[maxIndex]);

			//Replica a melhor solução para todas as threads
			//replicateSolution<<<tamGrid, tamBlock>>>(d_sol, d_rc, d_solValue, maxIndex);
			//CHECK_ERROR(cudaDeviceSynchronize());
			//printf("\nReplicou");

			//Atualiza a temperatura de acordo com o fator de controle
			//de temperatura
			//temp = factor * temp;
		//}//Fim do laço principal do SA

		//printf("\nSaí da cadeia de Markov");

		//Copia as soluções de volta para a CPU
		CHECK_ERROR(cudaMemcpy(h_solValue, d_solValue, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost));
		CHECK_ERROR(cudaMemcpy(h_sol, d_sol, numBlocks * numThreads * h_m->numItems * sizeof(int), cudaMemcpyDeviceToHost));
		//CHECK_ERROR(cudaMemcpy(h_solValue, d_solValue, numBlocks * numThreads * sizeof(int), cudaMemcpyDeviceToHost));
		//CHECK_ERROR(cudaDeviceSynchronize());

		maxIndex = findBestSolIndex(h_solValue);
		//bestSolThreadValue = h_bestSolValues[bestSolThreadIndex];

		cudaEventRecord(e_Stop, cudaEventDefault);
		cudaEventSynchronize(e_Stop);
		cudaEventElapsedTime(&elapsedTimeGPU, e_Start, e_Stop);
		totalTime = elapsedTimeGPU / 1000;

		//printf("\nAntes de escrever");

		//Escreve o resultado final da execução
		writeBestResult(fileOut, fileOutName, totalTime, exec, maxIndex);

		//printf("\nEscrevi");

		vetGaps[exec] = (double)((double)(optSol - h_solValue[maxIndex]) / optSol) * 100;
		vetTimes[exec] = totalTime;

		mediumTime += totalTime;
		mediumGap += vetGaps[exec];

		if (h_solValue[maxIndex] >= bestOverallSolValue)
		{
			bestOverallSolValue = h_solValue[maxIndex];
			bestOverallTime = totalTime;
			bestOverallGap = vetGaps[exec];
		}

		if (vetGaps[exec] == 0.0)
			numOpt++;
		else if (vetGaps[exec] < 0.0)
			numBetter++;

		//printf("Exec = %d ", exec);

	}//Fim das execuções

	//printf("\nTerminei!!!!");

	//Calcular média e dessvio padrão das execuções
	mediumTime /= EXECS;
	mediumGap /= EXECS;

	dpTimes = 0.0;
	dpGaps = 0.0;

	for (int i = 0; i < EXECS; i++)
	{
		dpTimes += pow((vetTimes[i] - mediumTime), 2);
		dpGaps += pow((vetGaps[i] - mediumGap), 2);
	}

	dpTimes /= EXECS;
	dpGaps /= EXECS;

	dpTimes = sqrt(dpTimes);
	dpGaps = sqrt(dpGaps);

	//printf("\nEscrita final!!!!");

	finalWrite(fileOut, fileOutName, mediumTime, mediumGap, dpTimes, dpGaps, bestOverallTime, bestOverallSolValue, numOpt);

	//printf("\nDepois Escrita final!!!!");

	//Libera a memória alocada
	free(h_m->values);
	free(h_m->useResources);
	free(h_m->resources);
	free(h_sol);
	//free(h_bestSol);
	//free(h_incSol);
	free(h_newSol);
	free(h_psUt);
	free(h_rc);
	//free(h_incRc);
	free(h_newRc);

	free(h_solValue);
	//free(h_incSolValue);

	free(vetGaps);
	free(vetTimes);

	CHECK_ERROR(cudaFree(d_sol));
	//CHECK_ERROR(cudaFree(d_bestSol));
	//CHECK_ERROR(cudaFree(d_incSol));
	CHECK_ERROR(cudaFree(d_newSol));
	CHECK_ERROR(cudaFree(d_rc));
	//CHECK_ERROR(cudaFree(d_incRc));
	CHECK_ERROR(cudaFree(d_newRc));
	CHECK_ERROR(cudaFree(d_psUt));
	CHECK_ERROR(cudaFree(d_solValue));
	//CHECK_ERROR(cudaFree(d_incSolValue));
	//CHECK_ERROR(cudaFree(d_newSolValue));
	CHECK_ERROR(cudaFree(d_rclIndexes));

	CHECK_ERROR(cudaFree(d_tested));

	//Retorna ao final do programa
	return 0;
}

__global__ void setupKernel(curandState* state, unsigned long seed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(seed, idx, 0, &state[idx]);
}

/*
 Função que constroi a solução inicial para a execução do Simulated Annealing.
 Cada thread a executa e obtém uma solução própria.
 */
__global__ void constructSolution(int* d_sol, int* d_solValues, int* d_rc, float* d_psUt,
								  float alpha, curandState* state, int* d_rclIndexes)
{
	//Memória compartilhada
	//extern __shared__ int rclIndexes[];
	int actualRclSize;
	int i, index, min, max, localIndex, line;
	float rclFloor;
	bool full;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Variável com o deslocamento inicial do início dos vetores globais
	int displItems = idx * c_numItems[0];
	int displResources = idx * c_numResources[0];
	int endItems = displItems + c_numItems[0];
	//int endResources = displResources + c_numResources[0];

	//Variável lógica que indica se a mochila já está cheia
	full = false;
		
	//printf("Bloco: %d, Displ: %d\n", blockIdx.x, displ);

	for (i = displItems; i < endItems; i++)
		d_sol[i] = 0;

	for (i = 0; i < c_numResources[0]; i++)
		d_rc[displResources + i] = tex1Dfetch(tex_resources, i);

	//Enquanto a mochila não estiver cheia
	while (!full)
	{
		//Calcula as pseudoutilidades dos itens na GPU.
		calculatePseudoUtilities(d_sol, d_psUt, d_rc);

		//Baseado no valor de alpha considera a RCL composta pelos
		//itens cuja pseudoutilidade entre os alpha% possíveis melhores valores
		//no intervalo entre o melhor e o pior item.
		max = getMaxPsUtIndex(d_psUt);
		min = getMinPsUtIndex(d_psUt);

		//printf("Máximo[%d]: %f Mínimo[%d]: %f", max, curPsUt[displ + max], min, curPsUt[displ + min]);

		rclFloor = d_psUt[max] - alpha * (d_psUt[max] - d_psUt[min]);

		//printf("Psut mínima: %f.\n", rclFloor);

		//Preenche a RCL com os índices dos candidatos		
		actualRclSize = 0;

		for (i = displItems; i < endItems; i++)
		{
			//Antes de acrescentar à RCL, verifica se o item já não está
			//na solução pois, nesse caso, ele não é um candidato à adição
			//int pos = curPsUtArray[displ + i].index % blockDim.x; //Armazena a posição dentro do vetor de solução local
			if (d_sol[i] == 0  && d_psUt[i] >= rclFloor)//Não está na solução
			{
				d_rclIndexes[displItems + actualRclSize] = i;//Armazena o índice na solução global
				actualRclSize++;
			}
		}

		//printf("RCL pronta[%d]: Tam: %d\n", blockIdx.x, actualRclSize);

		//printf("Alpha = %f - Tamanho RCL: %d.\n", alpha, actualRclSize);

		//Gera um número aleatório para escolher um dos elementos
		//contidos na RCL
		if (actualRclSize > 0)
		{
			index = curand(&state[idx]) % actualRclSize;//Gera número aleatório entre 0 e tamanho da RCL - 1

			index = d_rclIndexes[displItems + index];//O índice é relativo à indexação no vetor global

			//Obtém o índice local do subvetor do bloco em questão
			localIndex = index % c_numItems[0];

			//Tenta adicionar o item escolhido, verificando se ele não
			//torna a mochila inviável
			//Nessa implementação, se o primeiro não puder ser colocado,
			//considera-se a solução como completada
			//added = false;

			if (canAdd(localIndex, d_sol, d_rc))
			{
				//printf("Bloco %d Adiciona %d ", blockIdx.x, index);
				//Pode adicionar o item à solução local corrente

				d_sol[index] = 1;

				line = localIndex * c_numResources[0];
				for (i = 0; i < c_numResources[0]; i++)
					d_rc[displResources + i] -= tex1Dfetch(tex_useResources, line + i);
			}
			else
				full = true;
		}
		else
			full = true;
	}//Fim do while que verifica se a mochila ainda não está cheia
	//*/
	//Calcula o valor da solução
	d_solValues[idx] = getSolutionValue(d_sol);
	//printf("Calculou valor [%f]", curBestSolValues[blockIdx.x]);
	//fflush(stdin);
}

/*
 * Função que replica a melhor solução entre as threads
 */
__global__ void replicateSolution(int* d_sol, int* d_rc, int* d_solValue, int index)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int sourceDisplItems, sourceDisplResources, destDisplItems, destDisplResources;

	sourceDisplItems = index * c_numItems[0];
	sourceDisplResources = index * c_numResources[0];
	destDisplItems = idx * c_numItems[0];
	destDisplResources = idx * c_numResources[0];

	for (i = 0; i < c_numItems[0]; i++)
		d_sol[destDisplItems + i] = d_sol[sourceDisplItems + i];
	//printf(" Dn%d/%d ", idx, d_sol[idx]);

	for (i = 0; i < c_numResources[0]; i++)
		d_rc[destDisplResources + i] = d_rc[sourceDisplResources + i];

	d_solValue[idx] = d_solValue[index];
}

/*
 * Função que executa a etapa da cadeia de Markov do Simulated Annealing
 */
__global__ void markovChainKernel(int* sol, int* newSol, int* rc, int* newRc, int* solValue, curandState* devStates, int maxChainLength, 
								  int* d_tested, double initTemp, double finalTemp, float factor)
{
	//extern __shared__ int sh_tested[];
	int i, index, drop, add, globalIndex, newSolValue, line;
	float delta;
	bool changed, goon, found;
	curandState st;

	int chainLength;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Variável com o deslocamento inicial do início dos vetores globais
	int displItems = idx * c_numItems[0];
	int displResources = idx * c_numResources[0];
	int endItems = displItems + c_numItems[0];
	int endResources = displResources + c_numResources[0];

	st = devStates[idx];

/*	for (i = displItems; i < endItems; i++)
		newSol[i] = sol[i];

	for (i = displResources; i < endResources; i++)
		newRc[i] = rc[i];
*/
	//printf("\nT = %.6f\n", temp);
	double temp = initTemp;
	while (temp >= finalTemp)
	{
		for (chainLength = 0; chainLength < maxChainLength; chainLength++)
		{
			//printf("%d ", chainLength);
			//Escolhe um item aleatoriamente
			index = curand(&st) % c_numItems[0];
	
			globalIndex = displItems + index;
	
			changed = false;				
	
			//Se o item ainda não estiver na solução, é adicionado
			if (newSol[globalIndex] == 0)
			{
				//printf("Não");
				newSol[globalIndex] = 1;
	
				//Atualizar rc para refletir a adição do novo item
				line = index * c_numResources[0];
				for (i = 0; i < c_numResources[0]; i++)
					newRc[displResources + i] -= tex1Dfetch(tex_useResources, line + i);
	
				//Laço que retira itens para garantir que a solução volte a ser viável
				while (!isFeasibleRem(newRc))
				{
					//printf(".");
	
					//Coloquei esse laço aqui para garantir que o
					//item retirado é diferente do que acabamos de
					//adicionar e que ele realmente está na mochila
					do
					{
						//printf("-");
						drop = curand(&st) % c_numItems[0];
					} while(drop == index || newSol[displItems + drop] == 0);//Se o item for o mesmo que foi adicionado ou não estiver na mochila, a busca continua....
	
					//globalDrop = displItems + drop;
	
					newSol[displItems + drop] = 0;
	
					//Atualiza rc para refletir a retirada do item
					line = drop * c_numResources[0];
					for (i = 0; i < c_numResources[0]; i++)
						newRc[displResources + i] += tex1Dfetch(tex_useResources, line + i);
	
				}
	
				//printf("Drop: %d ", drop);
				newSolValue = getSolutionValue(newSol);
	
				delta = newSolValue - solValue[idx];
	
				if (delta >= 0 || ((double) curand_uniform(&st)) < ((double) exp((double) delta / temp)))
				{
					//Atualiza solução
					changed = true;
				}
			}
			else//Se o item já estava na solução, ele é retirado
			{
				//printf("Sim");
				newSol[globalIndex] = 0;
	
				//Outro item é adicionado aleatoriamente
				found = false;
	
				//Código potencialmente problemático!
				//E se ninguém puder ser adicionado?
				//Pode ser necessário estabelecer um limite para a
				//busca
				goon = true;
	
				//Atualiza rc
				line = index * c_numResources[0];
				for (i = 0; i < c_numResources[0]; i++)
					newRc[displResources + i] += tex1Dfetch(tex_useResources, line + i);
	
				//printf("Drop: %d ", index);
				//int* tested = (int*) malloc(d_m->numItems * sizeof(int));
				for (i = displItems; i < endItems; i++)
					d_tested[i] = 0;
	
				d_tested[displItems + index] = 1;
				
				while (!found && goon)//sModifiquei para tentar adicionar mais de um novo item
				{
					do
					{
						//printf("+");
						add = curand(&st) % c_numItems[0];
						d_tested[displItems + add] = 1;
					} while(add == index || newSol[displItems + add] == 1);
	
					//globalAdd = displItems + add;
	
					if (canAdd(add, newSol, newRc))
					{
						newSol[displItems + add] = 1;
	
						//Atualiza rc
						line = add * c_numResources[0];
						for (i = 0; i < c_numResources[0]; i++)
							newRc[displResources + i] -= tex1Dfetch(tex_useResources, line + i);
						
						//printf("Add: %d ", add);
	
						found = true;
					}
					//else
					//{
						if (allTested(d_tested))
							goon = false;		
					//}
				}		
				
				newSolValue = getSolutionValue(newSol);
	
				delta = newSolValue - solValue[idx];
	
				if (delta >=0 || ((double) curand_uniform(&st)) < ((double) exp((double) delta / temp)))
				{
					changed = true;
				}
			}//Fim if-else
	
			if (newSolValue > solValue[idx])
			//if (incSolValue[blockIdx.x] < solValue[blockIdx.x])
			{
				for (i = displItems; i < endItems; i++)
					sol[i] = newSol[i];
	
				for (i = displResources; i < endResources; i++)
					rc[i] = newRc[i];
	
				solValue[idx] = newSolValue;
			}
			//else
			if (!changed)
			{
				for (i = displItems; i < endItems; i++)
					newSol[i] = sol[i];
	
				for (i = displResources; i < endResources; i++)
					newRc[i] = rc[i];
			}
		}//Fim do for da cadeia de Markov
		
		temp = factor * temp;
	}
	devStates[idx] = st;
}

/*
 * Função que verifica se todas as posições de um vetor possuem o valor
 * diferente de 0
 */
__device__ bool allTested(int* d_tested)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Variável com o deslocamento inicial do início dos vetores globais
	int displItems = idx * c_numItems[0];
	int endItems = displItems + c_numItems[0];

	for (i = displItems; i < endItems; i++)
	{
		if (d_tested[i] == 0)
			return false;
	}

	return true;
}

/*
 * Função para inicialização de valores de capacidade residual da heurística KMW
 */
__global__ void initialize(int* d_rc)
{
	/*for (int i = 0; i < numBlocks; i++)
	{
		int displ = i * h_m->numResources;
		for (int j = 0; j < h_m->numResources; j++)
			h_rc[displ + j] = h_m->resources[j];
	}*/

	int j;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int displ = idx * c_numResources[0];
	for (j = 0; j < c_numResources[0]; j++)
		d_rc[displ + j] = tex1Dfetch(tex_resources, j);
}

/*
 * Função para cálculo das pseudoutilidades dos itens
 */
__device__ void calculatePseudoUtilities(int* curSol, float* curPsUt, int* curRc)
{
	int i, j, line;
	float sum;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Variáveis com o deslocamento inicial do início dos vetores globais
	int displItems = idx * c_numItems[0];
	int displResources = idx * c_numResources[0];
	//int localIdx = idx % d_m->numItems;

	for (i = 0; i < c_numItems[0]; i++)
	{
		if (curSol[displItems + i] == 0)
		{
			sum = 0.0;
			line = i * c_numResources[0];
			for (j = 0; j < c_numResources[0]; j++)
				sum += (float) tex1Dfetch(tex_useResources, line + j) / curRc[displResources + j];

			curPsUt[displItems + i] = (float) tex1Dfetch(tex_values, i) / sum;
		}
		else
			curPsUt[displItems + i] = 0.0;
	}
}

/*
 * Função para verificar se um elemento pode ser adicionado à solução atual.
 * O parâmetro index é o índice no vetor local de soluções.
 */
__device__ bool canAdd(int index, int* curSol, int* curRc)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displItems = idx * c_numItems[0];
	int displRc = idx * c_numResources[0];

	if (curSol[displItems + index] == 1)
		return false;

	//int localIndex = index % d_m->numItems;

	int line = index * c_numResources[0];
	for (i = 0; i < c_numResources[0]; i++)
	{
		if (curRc[displRc + i] - tex1Dfetch(tex_useResources, line + i) < 0)
			return false;
	}

	return true;
}

/*
 * Função para cálculo do valor da solução atual
 */
__device__ int getSolutionValue(int* curSol)
{
	int i;
	int value = 0;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displ = idx * c_numItems[0];

	for (i = 0; i < c_numItems[0]; i++)
		value += curSol[displ + i] * tex1Dfetch(tex_values, i);

	return value;
}

/*
 * Função para verificar se acrescentar o item index à solução atual é
 * possível
 */
__device__ bool isFeasibleAdd(int* d_rc, int index)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displResources = idx * c_numResources[0];

	int line = index * c_numResources[0];
	for (i = 0; i < c_numResources[0]; i++)
	{
		if (d_rc[displResources + i] - tex1Dfetch(tex_useResources, line + i) < 0)
			return false;
	}

	return true;
}

__device__ bool isFeasibleRem(int* curRc)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displResources = idx * c_numResources[0];
	int endResources = displResources + c_numResources[0];

	for (i = displResources; i < endResources; i++)
	{
		if (curRc[i] < 0)
			return false;
	}

	return true;
}

/*
 * Função que obtém e retorna o índice do elemento com a maior pseudoutilidade diferente de zero.
 * O índice retornado é aquele do vetor global de pseudoutilidades.
 */
__device__ int getMaxPsUtIndex(float* v)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displ = idx * c_numItems[0];
	int end = displ + c_numItems[0];
	
	int iMax = displ;

	for (i = displ + 1; i < end; i++)
	{

		if (v[i] != 0.0 && v[i] > v[iMax])
			iMax = i;
	}

	return iMax;
}

/*
 * Função que obtém e retorna o índice do elemento com a menor pseudoutilidade diferente de zero.
 * O índice retornado é aquele do vetor global de pseudoutilidades.
 */
__device__ int getMinPsUtIndex(float* v)
{
	int i;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int displ = idx * c_numItems[0];
	int end = displ + c_numItems[0];
	
	int iMin = displ;

	for (i = displ + 1; i < end; i++)
	{
		if (v[i] != 0.0 && v[i] < v[iMin])
			iMin = i;
	}

	return iMin;
}

/*
 * Função que escreve o melhor resultado de todas as iterações de uma
 * execução.
 */
__host__ void writeBestResult(FILE* f, char* nome, double time, int exec, int maxIndex)
{
	FILE* fGaps;
	FILE* fSols;
	FILE* fTimes;

    f = fopen(nome, "a+");
    fGaps = fopen("cudasa_allgaps.cu.out", "a+");
    fSols = fopen("cudasa_allsols.cu.out", "a+");
	fTimes = fopen("cudasa_alltimes.cu.out", "a+");

	float gap = (double)((double)(optSol - h_solValue[maxIndex]) / optSol) * 100;
	
    fprintf(f, "Execução: %d\nThread: %d\nTempo Total: %.4f", exec, maxIndex, time);
    fprintf(f, "\nSolução ótima = %d\nMelhor solução = %d\nGap: %.4f\n",
			optSol, h_solValue[maxIndex], gap);

    fprintf(fGaps,"%.4f\n", gap);
	fprintf(fSols,"%d\n", h_solValue[maxIndex]);
	fprintf(fTimes,"%.4f\n", time);
    fprintf(f, "Solução:\n");

    int displ = maxIndex * h_m->numItems;

    for (int i = 0; i < h_m->numItems; i++)
		fprintf(f, "%d", h_sol[displ + i]);

    fprintf(f, "\n\n");

    fclose(f);
    fclose(fGaps);
    fclose(fSols);
    fclose(fTimes);
}

/*
 * Função para escrever no arquivo de saída
 */
__host__ void finalWrite(FILE* f, char* nome, double mediumTime, double mediumGap, double dpTimes, double dpGaps,
                double bestOverralTime, int bestOverallSolValue, int numOpt)
{
	FILE* f2;
	double bestGap = (float)((float)(optSol - bestOverallSolValue) / optSol) * 100;

    f = fopen(nome, "a+");
    f2 = fopen("estartnovocudasa.out", "a+");

    fprintf(f, "Execuções: %d \nT0: %.6f \nTF: %.6f \nCadeia: %d \nFator temp: %.4f \nTempo Medio: %.4f \nGap Médio: %.4f \nDesvio-Padrão (Gaps): %.4f\nDesvio-Padrão (Tempos): %.4f\nMelhor Solução: %d, Melhor Gap: %.4f, Melhor Tempo: %.4f\nÓtimas: %d\nMelhores: %d\n",
			EXECS, initTemp, finalTemp, maxChainLength, factor, mediumTime, mediumGap, dpGaps, dpTimes, bestOverallSolValue, bestGap, bestOverallTime, numOpt, numBetter);

	fprintf(f2, "%d %.4f %.4f %.4f %.4f %.4f\n", numOpt, bestGap, mediumGap, dpGaps, mediumTime, dpTimes);
    //fprintf(f, "\n\n");

    fclose(f);
    fclose(f2);
}

/*
 * Função para achar o índice da melhor solução que as threads encontraram
 */
__host__ int findBestSolIndex(int* v)
{
	int i, max;

	max = 0;

	for (i = 1; i < numBlocks * numThreads; i++)
	{
		if (v[i] > v[max])
			max = i;
	}

	return max;
}
