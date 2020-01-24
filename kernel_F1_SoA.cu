#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>

#include <curand.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>


//#define PSIZE 10
//#define NGEN 500000
#define MUT_PROB 0.05

struct Individual 
{
    float fitness;
    unsigned int chromossomes;
};

struct Population
{
	float *fitness;
	unsigned int *chromossomes;
};


__host__ __device__ bool operator<(const Individual &i, const Individual &j)
{
    return (i.fitness > j.fitness);
}

__host__ __device__ float operator+(const Individual &i, const Individual &j)
{
    return (i.fitness + j.fitness);
}

__host__ __device__ Individual individualSum (Individual i, Individual j)
{
    Individual I;
    I.fitness = i.fitness + j.fitness;
    return I;
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

__global__ void createPopulation(Population *population, unsigned int seed, curandState_t *states)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);

    population->fitness[id] = 0;
    population->chromossomes[id] = curand(&states[id]);
}

__global__ void fitness(Population *population, float *totalFitness)
{
    int id = blockIdx.x;
    unsigned int mask = 0x3FF;
    __shared__ float a[3];

    a[threadIdx.x] = (population->chromossomes[id] & (mask << (10 * threadIdx.x))) >> (10 * threadIdx.x);

    a[threadIdx.x] = (a[threadIdx.x] - 512)/100.0;

    __syncthreads();

    if(threadIdx.x == 0)
    {
        float f = 1.0 / (1 + a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
        population->fitness[id] = f;
        atomicAdd(totalFitness, f);
    }
}

__global__ void reproduce(Population *population, Population *nextPopulation, int PSIZE, float *totalFitness, curandState_t *states)
{ 
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int parents[2];
    int temp = -1;
    float localTotalFitness = *totalFitness;
    //Selection
    for(int j = 0; j < 2; j++)
    {
        float p = curand_uniform(&states[id]) * localTotalFitness;
        float score = 0;

        for(int k = 0; k < PSIZE; k++)
        {
            if(k == temp)
            {
                continue;
            }
            score += population->fitness[k];
            if(p < score)
            {
                parents[j] = population->chromossomes[k];
                localTotalFitness -= population->fitness[k];
                temp = k;
                break;
            }
        }
    }

    //Crossover
    unsigned char cutPoint = curand(&states[id]) % 31;
    unsigned mask1 = 0xffffffff << cutPoint; 
    unsigned mask2 = 0xffffffff >> (32 - cutPoint);
    unsigned int child;
    child = (parents[0] & mask1) + (parents[1] & mask2);

    //Mutation
    float mutation = curand_uniform(&states[id]);
    if(mutation < MUT_PROB)
    {
        unsigned char mutPoint = curand(&states[id]) % 30;
        child ^= 1 << mutPoint;
    }



    nextPopulation->chromossomes[id] = child;
    nextPopulation->fitness[id] = 0;
    if(id == 0)
    {
        nextPopulation->chromossomes[0] = population->chromossomes[0];
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
        //printf("ANTES");
        clock_t start, end;
        //Init variables

        Population *population, *nextPopulation, *swap, *population_h, *nextPopulation_h;

        population_h = (Population *) malloc(sizeof(Population));
        nextPopulation_h = (Population *) malloc(sizeof(Population));

        cudaMalloc((void**) &population, sizeof(Population));
        cudaMalloc((void**) &nextPopulation, sizeof(Population));

        cudaMalloc((void**) &(population_h->fitness), PSIZE * sizeof(float));
        cudaMalloc((void**) &(population_h->chromossomes), PSIZE * sizeof(unsigned int));
        cudaMalloc((void**) &(nextPopulation_h->fitness), PSIZE * sizeof(float));
        cudaMalloc((void**) &(nextPopulation_h->chromossomes), PSIZE * sizeof(unsigned int));

        cudaMemcpy(population, population_h, sizeof(Population), cudaMemcpyHostToDevice);
        cudaMemcpy(nextPopulation, nextPopulation_h, sizeof(Population), cudaMemcpyHostToDevice);

        curandState_t *states;
        cudaMalloc((void**) &states, PSIZE * sizeof(curandState_t));


        float *maxFitness;
        maxFitness = (float *) malloc(NGEN * sizeof(float));


        float *totalFitness;
        cudaMalloc((void**) &totalFitness, sizeof(float));

	start = clock();

        //printf("marco0");
        //Create population
        createPopulation<<<ceil(PSIZE/1024.0), min(PSIZE, 1024)>>>(population, time(NULL), states);
        //printf("marco1");

        float const zero = 0.0f;
        for(int i = 0; i < NGEN; i++)
        {
            cudaMemcpy(totalFitness, &zero, sizeof(float), cudaMemcpyHostToDevice);
            //Calculate fitness
            fitness<<<PSIZE, 3>>>(population, totalFitness);
            //printf("marco2");

            //thrust::device_ptr<Population> dev_ptr_population(population);
            //thrust::sort(dev_ptr_population, dev_ptr_population + PSIZE);
            thrust::device_ptr<float> dev_ptr_fitness(population_h->fitness);
            thrust::device_ptr<unsigned int> dev_ptr_chromossomes(population_h->chromossomes);
            thrust::sort_by_key(dev_ptr_fitness, dev_ptr_fitness + PSIZE, dev_ptr_chromossomes, thrust::greater<float>());
            cudaMemcpy(&maxFitness[i], &(population_h->fitness[0]), sizeof(float), cudaMemcpyDeviceToHost);

            //printf("marco3");
            reproduce<<<ceil(PSIZE/1024.0), min(PSIZE, 1024)>>>(population, nextPopulation, PSIZE, totalFitness, states);
        //printf("marco4");
                
            swap = population;
            population = nextPopulation;
            nextPopulation = swap;

            swap = population_h;
            population_h = nextPopulation_h;
            nextPopulation_h = swap;
        }
  
        end = clock();
        cudaFree(population);
        cudaFree(nextPopulation);
        cudaFree(states);

        if(PRINT != 0)
        {   
            printf("Gen\tFitness\n");
            for(int i = 0; i < NGEN; i++)
            {
                printf("%d\t%f\n", i+1, maxFitness[i]);
            }
        }

        free(maxFitness);

        printf("\nT total(us)\t\tT geração(us)\n");
        double cpu_time_used = 1000000 * ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("%f\t\t%f\n\n", cpu_time_used, cpu_time_used/NGEN);
        Ttotal += cpu_time_used;
    }

    printf("\nAvg T total(us)\t\tAvg T geração(us)\n");
    printf("%f\t\t%f\n", Ttotal/NIT, Ttotal/(NIT*NGEN));
    
return 0;
}
