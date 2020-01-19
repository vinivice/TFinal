#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>


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


__global__ void persistentThreads(int popSize, int NGEN, float *maxFitness, unsigned int seed)
{
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
        for(int j = 0; j < CHROMO_SIZE; j++)
        {
            population[i].chromossomes[j] = curand(&state)%256 - 128;
        }
    }
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
                unsigned char cutPoint = curand(&state) % (CHROMO_SIZE + 1);
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
                float mutation = curand_uniform(&state);
                if(mutation < MUT_PROB)
                {
                    int mutPoint = curand(&state) % CHROMO_SIZE;
                    child.chromossomes[mutPoint] = curand(&state) % 256 - 128;
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

        start = clock();

        persistentThreads<<<1, min(PSIZE, 1024), PSIZE * sizeof(Individual)>>>(PSIZE, NGEN, maxFitness, time(NULL));

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

