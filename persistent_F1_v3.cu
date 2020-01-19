

#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <cooperative_groups.h>


//#define PSIZE 10
//#define NGEN 500000
#define MUT_PROB 0.05
#define TESTE 256

struct Individual 
{
    float fitness;
    unsigned int chromossomes;
};

__device__ bool comparator (Individual i, Individual j)
{
    return (i.fitness > j.fitness);
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

__device__ int lock = 0;

/*__device__ void lockBlocks(int goal)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    __syncthreads();
    if(threadIdx.x == 0)
    {
//	printf("\nPRE_ADD\tlock: %d\tBlocks: %d\tid: %d", lock, goal, id);
        atomicAdd(&lock, 1);
//	printf("\nPOS_ADD\tlock: %d\tBlocks: %d\tid: %d", lock, goal, id);
        while(lock != 0)
        {
           printf("A");
           if(id == 0 && lock == goal)
            {
//                printf("BB");
		lock = 0;
            }
        }
  //  printf("\nlock: %d\tBlocks: %d\tid: %d", lock, goal, id);
    }
    __syncthreads();

}*/


__global__ void persistentThreads(int popSize, int NGEN, float *maxFitness, unsigned int seed, Individual *population, int numBlocks)
//__global__ void persistentThreads(int popSize, int NGEN, unsigned int seed, Individual *population, int numBlocks)
{
using namespace cooperative_groups;
//    printf("HERE");
grid_group grid = this_grid();
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Individual child;
    curandState_t state;
    curand_init(seed, id, 0, &state);

    __shared__ float totalFitness;
    //extern __shared__ Individual population[];


    for(int i = id; i < popSize; i+=blockDim.x * numBlocks)
    {
        //Create population
        population[i].fitness = 0;
        population[i].chromossomes = curand(&state);
    }
grid.sync();

//    lockBlocks(numBlocks); 

    //printf("\nlock: %d\tBlocks: %d\tid: %d", lock, numBlocks, id);
    
    for(int g = 0; g < NGEN; g++)
    {
        if(id == 0)
        {
            totalFitness = 0;
        }
grid.sync();

// lockBlocks(numBlocks);

        for(int i = id; i < popSize; i+=blockDim.x * numBlocks)
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
grid.sync();

// lockBlocks(numBlocks);

//printf("C");
        if(id == 0)
        {
            thrust::sort(population, population + popSize, comparator);
	    maxFitness[g] = population[0].fitness;
	    //printf("%d\t%f\n", g, population[0].fitness);
	}
grid.sync();

// lockBlocks(numBlocks);


        
        float localTotalFitness = totalFitness;

        for(int i = id; i < popSize; i+=blockDim.x * numBlocks)
        {

            Individual parents[2];
            int temp = -1;

            //Selection
            for(int j = 0; j < 2; j++)
            {
                float p = curand_uniform(&state) * localTotalFitness;
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
            unsigned char cutPoint = curand(&state) % 31;
            unsigned int mask1 = 0xffffffff << cutPoint; 
            unsigned int mask2 = 0xffffffff >> (32 - cutPoint);
            child.fitness = 0;
            child.chromossomes = (parents[0].chromossomes & mask1) + (parents[1].chromossomes & mask2);
 
            //Mutation
            float mutation = curand_uniform(&state);
            if(mutation < MUT_PROB)
            {
                unsigned char mutPoint = curand(&state) % 30;
                child.chromossomes ^= 1 << mutPoint;
            }
        }
grid.sync();


// lockBlocks(numBlocks);


        
        if(id == 0)
        {
            child = population[0];
        }

        for(int i = id; i < popSize; i+=blockDim.x)
        {
            population[i] = child;
        }
grid.sync();

// lockBlocks(numBlocks);


        
    }
}

int main(int argc, char *argv[ ]) 
{
    using namespace cooperative_groups;
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

        Individual *population;
        cudaMalloc((void**) &population, PSIZE * sizeof(Individual));

 //       int *lock;
   //     cudaMalloc((void**) &lock, sizeof(int));

        int numBlocks = 1152;
        int numThreads = 32;


 //       lockBlocks<<<numBlocks, numThreads>>>(numBlocks);


        CUdevice dev;
        cuDeviceGet(&dev,0); 

        cudaDeviceProp deviceProp;
        int numBlocksPerSm;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, persistentThreads, numThreads, 0);

        time_t t = time(NULL);
        void **args = ((void **) malloc(6 * sizeof(void *)));
        args[0] = &PSIZE;
        args[1] = &NGEN;
        args[2] = &maxFitness;
        args[3] = &t;
        args[4] = &population;
        args[5] = &numBlocks;

//	printf("Gen\tFitness\n");

	start = clock();

        cudaLaunchCooperativeKernel((void*)persistentThreads, deviceProp.multiProcessorCount*numBlocksPerSm, numThreads, args);


//        persistentThreads<<<numBlocks, numThreads>>>(PSIZE, NGEN, maxFitness, time(NULL), population, numBlocks);
        //persistentThreads<<<1, min(PSIZE, 1024)>>>(PSIZE, NGEN, maxFitness, time(NULL), population);
	
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

