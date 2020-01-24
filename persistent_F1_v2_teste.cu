#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>


//#define PSIZE 10
//#define NGEN 500000
#define MUT_PROB 0.05
#define TESTE 256
//#define VERSION 2

struct Individual 
{
    float fitness;
    unsigned int chromossomes;
};

__device__ bool comparator (Individual i, Individual j)
{
    return (i.fitness > j.fitness);
}

__host__ __device__ Individual operator+(const Individual &i, const Individual &j)
{
	Individual k;
    k.fitness = (i.fitness + j.fitness);
	return k;
}


void printPop(Individual *population, int popSize, int print)
{
    if(print != 0)
    {
        for(int i = 0; i < popSize; i++)
        {
            printf("%f - ", population[i].fitness);
        }
    }
}

__global__ void persistentThreads(int popSize, int NGEN, float *maxFitness, unsigned int seed, unsigned int *randomChromossomes, float *randomThresholds)
{
    volatile float forcaLoop;
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Individual child;
    curandState_t state;
    curand_init(seed, id, 0, &state);

    __shared__ float totalFitness;
    extern __shared__ Individual population[];

    for(int i = id; i < popSize; i+=blockDim.x)
    {
        //Create population
        population[i].fitness = 0;
        //population[i].chromossomes = curand(&state);
        population[i].chromossomes = randomChromossomes[2 * NGEN * popSize + threadIdx.x];
    }
    __syncthreads();


    for(int g = 0; g < NGEN; g++)
    {
	forcaLoop = 0;
        if(id == 0)
        {
            totalFitness = 0;
        }
        __syncthreads();
        for(int i = id; i < popSize; i+=blockDim.x)
        {
            //Calculate fitness
            
            unsigned int mask = 0x3FF;
            float a = 0, b = 0, c = 0;
            a = population[i].chromossomes & mask;
            b = (population[i].chromossomes & (mask << 10)) >> 10;
            c = (population[i].chromossomes & (mask << 20)) >> 20;

            a = (a - 512)/100.0;
            b = (b - 512)/100.0;
            c = (c - 512)/100.0;

            population[i].fitness = 1.0 / (1 + a*a + b*b + c*c);
            atomicAdd(&totalFitness, population[i].fitness);
        }
        __syncthreads();
        if(id == 0)
        {
            thrust::sort(population, population + popSize, comparator);
            maxFitness[g] = population[0].fitness;
            #if VERSION == 1
            Individual  teste;
            for(int zz = 0; zz < 10000; zz++)
	    {
		    teste.fitness = 0;
	    	for (int z = 0; z < popSize; z++)
            	{
                	teste = teste + population[z];
            	}
		forcaLoop += teste.fitness;
	    }            
	    ///printf("AAA%f\t%f\n", totalFitness, teste.fitness);
            #elif VERSION == 2
            Individual teste;
		thrust::device_ptr<Individual> ptr1(population);
		thrust::device_ptr<Individual> ptr2(population+popSize);
            for(int zz = 0; zz < 10000; zz++)
	    {
		    teste.fitness = 0;
		    teste = thrust::reduce(thrust::device, ptr1, ptr2, teste);
		forcaLoop += teste.fitness;
	    }
	    //printf("BBB%f\t%f\n", totalFitness, teste.fitness);
	    #endif
        }
        __syncthreads();
        
        float localTotalFitness = totalFitness;

        for(int i = id; i < popSize; i+=blockDim.x)
        {

            Individual parents[2];
            int temp = -1;

            //Selection
            for(int j = 0; j < 2; j++)
            {
                //float p = curand_uniform(&state) * localTotalFitness;
                float p = randomThresholds[3*popSize*blockDim.x*blockIdx.x + j*popSize + threadIdx.x] * localTotalFitness;
                float score = 0;

                for(int k = 0; k < popSize; k++)
                {
                    if(k == temp)
                    {
                        continue;
                    }
                    score += population[k].fitness;
                    if(p < score)
                    {
                        parents[j] = population[k];
                        localTotalFitness -= population[k].fitness;
                        temp = k;
                        break;
                    }
                }
            }

            //Crossover
            //unsigned char cutPoint = curand(&state) % 31;
            unsigned char cutPoint = randomChromossomes[2 * popSize * g +  threadIdx.x] % 31;
            unsigned int mask1 = 0xffffffff << cutPoint; 
            unsigned int mask2 = 0xffffffff >> (32 - cutPoint);
            child.fitness = 0;
            child.chromossomes = (parents[0].chromossomes & mask1) + (parents[1].chromossomes & mask2);
 
            //Mutation
            //float mutation = curand_uniform(&state);
            float mutation = randomThresholds[3*popSize*blockDim.x*blockIdx.x + 2*popSize + threadIdx.x];
            if(mutation < MUT_PROB)
            {
                //unsigned char mutPoint = curand(&state) % 30;
                unsigned char mutPoint = randomChromossomes[2 * popSize * g + popSize + threadIdx.x] % 30;
                child.chromossomes ^= 1 << mutPoint;
            }
        }
        __syncthreads();
        
        if(id == 0)
        {
            child = population[0];
        }

        for(int i = id; i < popSize; i+=blockDim.x)
        {
            population[i] = child;
        }
        __syncthreads();
        
    }
}

__global__ void generateRandomChromossomes(int popSize, int nGen, unsigned int *randomChromossomes, unsigned int seed)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(seed, id, 0, &state);
    
    //threadIdx.x ==> 1024
    //blockIdx.x ==> ceil((2*NGEN + 1) / 1024)

    for(int i = 0; i < popSize; i++)
    {
        int index = blockDim.x*(blockIdx.x * popSize + i) + threadIdx.x;
        if(index < (2 * nGen + 1) * popSize)
        {
            randomChromossomes[index] = curand(&state);
        }
    }

}
__global__ void generateRandomThresholds(int popSize, int nGen, float *randomThresholds, unsigned int seed)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(seed, id, 0, &state);
    
    //threadIdx.x ==> 1024
    //blockIdx.x ==> ceil((NGEN ) / 1024)

    for(int i = 0; i < popSize; i++)
    {
        int index = 3 * blockDim.x * (blockIdx.x * popSize + i) + threadIdx.x;
        if(index < 3 * nGen * popSize)
        {
            randomThresholds[index] = curand_uniform(&state);
            randomThresholds[index + 1024] = curand_uniform(&state);
            randomThresholds[index + 2048] = curand_uniform(&state);
        }
    }

}

int main(int argc, char *argv[ ]) 
{
    int PSIZE, NGEN, NIT, PRINT;
    double Ttotal = 0;
    if(argc < 5)
    {
        printf("Uso %s <POP_SIZE> <N_GEN> <N_ITERACOES> <PRINT>\n", argv[0]);
        return 1;
    }
    else
    {
        PSIZE = atoi(argv[1]);
        NGEN = atoi(argv[2]);
        NIT = atoi(argv[3]);
        PRINT = atoi(argv[4]);
    }

    for(int it = 0; it < NIT; it++)
    {   
        clock_t start, end;
        
        float *maxFitness, *cpu_maxFitness;
        cudaMalloc((void**) &maxFitness, NGEN * sizeof(float));
        cpu_maxFitness = (float *) malloc(NGEN * sizeof(float));

        //Random values
        unsigned int *randomChromossomes;
        cudaMalloc((void**) &randomChromossomes, ((2 * NGEN + 1 ) * PSIZE) * sizeof(unsigned int)); //One for creation and two for each generation
        generateRandomChromossomes<<<(((2 * NGEN) / 1024) + 1), 1024>>>(PSIZE, NGEN, randomChromossomes, time(NULL));
 //       generateRandomChromossomes<<<1, 1024>>>(PSIZE, NGEN, randomChromossomes, time(NULL));


        float *randomThresholds;
        cudaMalloc((void**) &randomThresholds, (3 * NGEN * PSIZE) * sizeof(float)); //two for the parents and one for mutation
        generateRandomThresholds<<<(((NGEN - 1) / 1024) + 1), 1024>>>(PSIZE, NGEN, randomThresholds, time(NULL));

	start = clock();

        persistentThreads<<<1, min(PSIZE, 1024), PSIZE * sizeof(Individual)>>>(PSIZE, NGEN, maxFitness, time(NULL), randomChromossomes, randomThresholds);

	cudaDeviceSynchronize();

	end = clock();

        cudaMemcpy(cpu_maxFitness, maxFitness, NGEN * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(maxFitness);
        if(PRINT != 0)
        {
            printf("Gen\tFitness\n");
            for(int i = 0; i < NGEN; i++)
            {
                printf("%d\t%f\n", i, cpu_maxFitness[i]);
            }
        }
        free(cpu_maxFitness);


        printf("\nT total(us)\t\tT geração(us)\n");
        double cpu_time_used = 1000000 * ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("%f\t\t%f\n\n", cpu_time_used, cpu_time_used/NGEN);
        Ttotal += cpu_time_used;

    }

    printf("\nAvg T total(us)\t\tAvg T geração(us)\n");
    printf("%f\t\t%f\n", Ttotal/NIT, Ttotal/(NIT*NGEN));
    
return 0;
}

