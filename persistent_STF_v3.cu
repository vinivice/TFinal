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
#define CHROMO_SIZE 256

struct Individual 
{
    float fitness;
    char chromossomes[CHROMO_SIZE];
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


__global__ void persistentThreads(int popSize, int NGEN, float *maxFitness, unsigned int seed, unsigned int *randomChromossomes, float *randomThresholds, Individual *population, int numBlocks)
{
    using namespace cooperative_groups;
    grid_group grid = this_grid();
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Individual child;
    curandState_t state;
    curand_init(seed, id, 0, &state);

    __shared__ float totalFitness;
    //extern __shared__ Individual population[];

    for(int i = id; i < popSize; i+=blockDim.x)
    {
        //Create population
        population[i].fitness = 0;
        for(int j = 0; j < CHROMO_SIZE; j++)
        {
            //population[i].chromossomes[j] = curand(&state)%256 - 128;
            population[i].chromossomes[j] = randomChromossomes[(3 * NGEN + j) * popSize + threadIdx.x]%256 - 128;
        }
    }
    grid.sync();
    __syncthreads();

    for(int g = 0; g < NGEN; g++)
    {
        if(id == 0)
        {
            totalFitness = 0;
        }
        __syncthreads();
        for(int i = id; i < popSize; i+=blockDim.x)
        {
            //Calculate fitness

            float subTotal = 0;
            for(int j = 0; j < CHROMO_SIZE; j++)
            {
                float x = (population[i].chromossomes[j] / 128.0) * 5.0;
                subTotal += (x*x*x*x - 16.0*x*x + 5.0*x) / 2.0;
            }
            population[i].fitness = 1.0 / (40.0*CHROMO_SIZE + subTotal);

            atomicAdd(&totalFitness, population[i].fitness);
        }
        __syncthreads();
        if(id == 0)
        {
            thrust::sort(population, population + popSize, comparator);
            maxFitness[g] = population[0].fitness;
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
                //unsigned char cutPoint = curand(&state) % (CHROMO_SIZE + 1);
                unsigned char cutPoint = randomChromossomes[3
 * popSize * g + threadIdx.x] % (CHROMO_SIZE + 1);
                child.fitness = 0;
                for(int j = 0; j < cutPoint; j++)
                {
                    child.chromossomes[j] = parents[0].chromossomes[j];
                }

                for(int j = cutPoint; j < CHROMO_SIZE; j++)
                {
                    child.chromossomes[j] = parents[1].chromossomes[j];
                }

                //Mutation
                //float mutation = curand_uniform(&state);
                float mutation = randomThresholds[3*popSize*blockDim.x*blockIdx.x + 2*popSize + threadIdx.x];
                if(mutation < MUT_PROB)
                {
                    //int mutPoint = curand(&state) % CHROMO_SIZE;
                    //child.chromossomes[mutPoint] = curand(&state) % 256 - 128;
                    int mutPoint = randomChromossomes[3 * popSize * g + popSize + threadIdx.x] % CHROMO_SIZE;
                    child.chromossomes[mutPoint] = randomChromossomes[3 * popSize * g + 2 * popSize + threadIdx.x] % 256 - 128;
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

__global__ void generateRandomChromossomes(int popSize, int nGen, int chromoSize, unsigned int *randomChromossomes, unsigned int seed)
{   
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(seed, id, 0, &state);
    
    //threadIdx.x ==> 1024
    //blockIdx.x ==> ceil((3*NGEN + chromoSize) / 1024)
    
    for(int i = 0; i < popSize; i++)
    {
        int index = blockDim.x*(blockIdx.x * popSize + i) + threadIdx.x;
        if(index < (3 * nGen + chromoSize) * popSize)
        {   
            randomChromossomes[index] = curand(&state);
        }
    }

}


__global__ void generateRandomThresholds(int popSize, int nGen, float *randomThresholds, unsigned int seed)
{   
    using namespace cooperative_groups;
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
        cudaMalloc((void**) &randomChromossomes, ((3 * NGEN + CHROMO_SIZE) * PSIZE) * sizeof(unsigned int)); //CHEOMO_SIZE for creation and three for each generation
        generateRandomChromossomes<<<(((3 * NGEN + CHROMO_SIZE - 1) / 1024) + 1), 1024>>>(PSIZE, NGEN, CHROMO_SIZE, randomChromossomes, time(NULL));
        //generateRandomChromossomes<<<1, 1024>>>(PSIZE, NGEN, randomChromossomes, time(NULL));


        float *randomThresholds;
        cudaMalloc((void**) &randomThresholds, (3 * NGEN * PSIZE) * sizeof(float)); //two for the parents and one for mutation
        generateRandomThresholds<<<(((NGEN - 1) / 1024) + 1), 1024>>>(PSIZE, NGEN, randomThresholds, time(NULL));


        Individual *population;
        cudaMalloc((void**) &population, PSIZE * sizeof(Individual));

        int numBlocks = 1152;
        int numThreads = 32;

        CUdevice dev;
        cuDeviceGet(&dev,0); 

        cudaDeviceProp deviceProp;
        int numBlocksPerSm;
        cudaGetDeviceProperties(&deviceProp, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, persistentThreads, numThreads, 0);

        time_t t = time(NULL);
        void **args = ((void **) malloc(8 * sizeof(void *)));
        args[0] = &PSIZE;
        args[1] = &NGEN;
        args[2] = &maxFitness;
        args[3] = &t;
        args[4] = &randomChromossomes;
        args[5] = &randomThresholds;
        args[6] = &population;
        args[7] = &numBlocks;


        start = clock();

        cudaLaunchCooperativeKernel((void*)persistentThreads, deviceProp.multiProcessorCount*numBlocksPerSm, numThreads, args);



        //persistentThreads<<<1, min(PSIZE, 1024), PSIZE * sizeof(Individual)>>>(PSIZE, NGEN, maxFitness, time(NULL), randomChromossomes, randomThresholds);

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

