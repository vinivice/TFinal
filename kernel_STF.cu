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
#define CHROMO_SIZE 256

struct Individual 
{
    float fitness;
    char chromossomes[CHROMO_SIZE];
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

__global__ void createPopulation(Individual *population, unsigned int seed, curandState_t *states)
{
    int id = blockIdx.x;
    int rand_id = blockIdx.x*blockDim.x + threadIdx.x;
    int c_id = threadIdx.x;

    curandState_t state;
    curand_init(seed, rand_id, 0, &state);
   
   
    
    population[id].chromossomes[c_id] = curand(&state)%256 - 128;

    population[id].fitness = 0;
    curand_init(seed, id, 0, &states[id]);
}

__global__ void fitness(Individual *population, float *totalFitness)
{
    int id = blockIdx.x;
    int c_id = threadIdx.x;
    __shared__ float a[CHROMO_SIZE];

    float x = (population[id].chromossomes[c_id] / 128.0) * 5.0;
    a[threadIdx.x] = (x*x*x*x - 16.0*x*x + 5.0*x) / 2.0;

    __syncthreads();

    if(threadIdx.x == 0)
    {
        float subTotal = thrust::reduce(thrust::device, a, a + CHROMO_SIZE, 0);
        float f = 1.0 / (40.0*CHROMO_SIZE + subTotal);
        population[id].fitness = f;
        atomicAdd(totalFitness, f);
    }
}

__global__ void reproduce(Individual *population, Individual *nextPopulation, int PSIZE, float *totalFitness, curandState_t *states)
{ 
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Individual parents[2];
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
    unsigned char cutPoint = curand(&states[id]) % (CHROMO_SIZE + 1);
    Individual child;
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
    float mutation = curand_uniform(&states[id]);
    if(mutation < MUT_PROB)
    {
        int mutPoint = curand(&states[id]) % CHROMO_SIZE;
        child.chromossomes[mutPoint] = curand(&states[id]) % 256 - 128;
    }


    nextPopulation[id] = child;
    if(id == 0)
    {
        nextPopulation[0] = population[0];
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
        //Init variables

        Individual *population, *nextPopulation, *swap;
        cudaMalloc((void**) &population, PSIZE * sizeof(Individual));
        cudaMalloc((void**) &nextPopulation, PSIZE * sizeof(Individual));


        curandState_t *states;
        cudaMalloc((void**) &states, PSIZE * sizeof(curandState_t));


        float *maxFitness;
        maxFitness = (float *) malloc(NGEN * sizeof(float));


        float *totalFitness;
        cudaMalloc((void**) &totalFitness, sizeof(float));

	start = clock();

        //Create population
        createPopulation<<<PSIZE, CHROMO_SIZE>>>(population, time(NULL), states);

        float const zero = 0.0f;
        for(int i = 0; i < NGEN; i++)
        {
            cudaMemcpy(totalFitness, &zero, sizeof(float), cudaMemcpyHostToDevice);
            //Calculate fitness
            fitness<<<PSIZE, CHROMO_SIZE>>>(population, totalFitness);

            thrust::device_ptr<Individual> dev_ptr_population(population);
            thrust::sort(dev_ptr_population, dev_ptr_population + PSIZE);
            cudaMemcpy(&maxFitness[i], &(population[0].fitness), sizeof(float), cudaMemcpyDeviceToHost);

            reproduce<<<ceil(PSIZE/1024.0), min(PSIZE, 1024)>>>(population, nextPopulation, PSIZE, totalFitness, states);
                
            swap = population;
            population = nextPopulation;
            nextPopulation = swap;
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
