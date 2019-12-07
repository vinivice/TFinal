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

struct Individual 
{
    float fitness;
    unsigned int chromossomes;
};

bool comparator (Individual i, Individual j)
{
    return (i.fitness > j.fitness);
}

Individual individualSum (Individual i, Individual j)
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
    curand_init(seed, id, 0, &states[id]);

    population[id].fitness = 0;
    population[id].chromossomes = curand(&states[id]);
}

__global__ void fitness(Individual *population)
{
    int id = blockIdx.x;
    unsigned int mask = 0x3FF;
    __shared__ float a[3];
        
    a[threadIdx.x] = (population[id].chromossomes & (mask << (9 * threadIdx.x))) >> (9 * threadIdx.x);

    a[threadIdx.x] = (a[threadIdx.x] - 512)/100.0;

    __syncthreads();

    if(threadIdx.x == 0)
    {
        population[id].fitness = 1.0 / (1 + a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
    }
}

__global__ void reproduce(Individual *population, Individual *nextPopulation, int PSIZE, float totalFitness, curandState_t *states)
{ 
    int id = blockIdx.x;
    Individual parents[2];
    int temp = -1;
    float localTotalFitness = totalFitness;
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
    unsigned char cutPoint = curand(&states[id]) % 28;
    unsigned mask1 = 0xffffffff << cutPoint; 
    unsigned mask2 = 0xffffffff >> (32 - cutPoint);
    Individual child;
    child.fitness = 0;
    child.chromossomes = (parents[0].chromossomes & mask1) + (parents[1].chromossomes & mask2);

    //Mutation
    float mutation = curand_uniform(&states[id]);
    if(mutation < MUT_PROB)
    {
        unsigned char mutPoint = curand(&states[id]) % 27;
        child.chromossomes ^= 1 << mutPoint;
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
        start = clock();
        //Init variables

        Individual *population, *nextPopulation, *swap;
        cudaMalloc((void**) &population, PSIZE * sizeof(Individual));
        cudaMalloc((void**) &nextPopulation, PSIZE * sizeof(Individual));


        curandState_t *states;
        cudaMalloc((void**) &states, PSIZE * sizeof(curandState_t));



/*
        Individual *population, *nextPopulation, *swap;
        population = (Individual *) malloc(PSIZE * sizeof(Individual));
        nextPopulation = (Individual *) malloc(PSIZE * sizeof(Individual));
*/
        float *maxFitness;
        maxFitness = (float *) malloc(NGEN * sizeof(float));


        float totalFitness = 0;


        //Create population
        createPopulation<<<PSIZE, 3>>>(population, time(NULL), states);

/*        for(int i = 0; i < PSIZE; i++)
        {
            population[i].fitness = 0;
            population[i].chromossomes = random();
        }*/
    
        //printPop(population, PSIZE, PRINT);
   
        for(int i = 0; i < NGEN; i++)
        {
            //Calculate fitness
            fitness<<<PSIZE, 3>>>(population);

//            totalFitness = fitness(population, PSIZE);
//            std::sort(population, population + PSIZE, comparator);
            thrust::sort(population, population + PSIZE, comparator);
            cudaMemcpy(&maxFitness[i], &(population[0].fitness), sizeof(float), cudaMemcpyDeviceToHost);
            //maxFitness[i] = population[0].fitness;
            totalFitness = thrust::reduce(population, population + PSIZE, population[0], individualSum).fitness;

            
            reproduce<<<PSIZE, 3>>>(population, nextPopulation, PSIZE, totalFitness, states);
                
            swap = population;
            population = nextPopulation;
            nextPopulation = swap;
        }
/*
        fitness(population, PSIZE);
        std::sort(population, population + PSIZE, comparator);

        //printf("\n");
        //printPop(population, PSIZE, PRINT);
        
        //printf("%f;%x", population[0].fitness, population[0].chromossomes);
        //printf("\n");
     */   
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
