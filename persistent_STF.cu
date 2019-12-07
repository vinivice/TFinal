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

/*__device__ float fitness(Individual *population, int id)
{
    unsigned int mask = 0x3FF;
    float a = 0, b = 0, c = 0;
    a = population[id].chromossomes & mask;
    b = (population[id].chromossomes & (mask << 9)) >> 9;
    c = (population[id].chromossomes & (mask << 18)) >> 18;

    a = (a - 512)/100.0;
    b = (b - 512)/100.0;
    c = (c - 512)/100.0;

    population[id].fitness = 1.0 / (1 + a*a + b*b + c*c);
}*/




//__global__ void persistentThreads(Individual *population, Individual *nextPopulation, int popSize, int NGEN, float *maxFitness, unsigned int seed)
__global__ void persistentThreads(int popSize, int NGEN, float *maxFitness, unsigned int seed)
{
 //   printf("KERNEL\n");
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    Individual child;
    curandState_t state;
    curand_init(seed, id, 0, &state);
 //   curand_init(0, id, 0, &state);
    //numbers[id] = curand(&state) % 100;

  //  Individual *swap;
//    population = pop;
  //  nextPopulation = nextPop;
    __shared__ float totalFitness;
    extern __shared__ Individual population[];

    //printf("%d ========== ", popSize);

    for(int i = id; i < popSize; i+=blockDim.x)
    {
        //Create population
        population[i].fitness = 0;
        for(int j = 0; j < CHROMO_SIZE; j++)
        {
            population[i].chromossomes[j] = curand(&state)%256 - 128;
        }
        //printf("%x - ", population[i].chromossomes);
    }
    __syncthreads();

    /*if(id == 0)
    {  
        printf("LALAAL\n");
        for(int i = 0; i < popSize; i++)
        {
            printf("%f ; %x\n", population[i].fitness, population[i].chromossomes);
        }
        printf("LALAAL\n");
    }*/

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
 //           (*nextPopulation)[0] = (*population)[0];
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
         //   printf("PA: %x\n", population[1].chromossomes);
           // printf("NPA: %x\n", nextPopulation[1].chromossomes);

           // nextPopulation[0] = population[0];
            child = population[0];
            //(*population) = NULL;

 //           printf("PD: %x\n", population[1].chromossomes);
   //         printf("NPD: %x\n", nextPopulation[1].chromossomes);
        }

        for(int i = id; i < popSize; i+=blockDim.x)
        {
            //population[i] = nextPopulation[i];
            population[i] = child;
        }
        __syncthreads();
        
    }
/*
    for(int iii = id; iii < popSize; iii+=blockDim.x)
    {
        //Calculate fitness
        
        unsigned int mask = 0x3FF;
        float a = 0, b = 0, c = 0;
        a = population[iii].chromossomes & mask;
        b = (population[iii].chromossomes & (mask << 9)) >> 9;
        c = (population[iii].chromossomes & (mask << 18)) >> 18;

        a = (a - 512)/100.0;
        b = (b - 512)/100.0;
        c = (c - 512)/100.0;

        population[iii].fitness = 1.0 / (1 + a*a + b*b + c*c);
        atomicAdd(&totalFitness, population[iii].fitness);
    }
    __syncthreads();
    if(id == 0)
    {
        thrust::sort(population, population + popSize, comparator);
//           (*nextPopulation)[0] = (*population)[0];
    }
*/
}

/*

int main()
{
  unsigned char *cpu_nums;
  cpu_nums = (unsigned char *) malloc(TESTE * sizeof(unsigned char));

  unsigned char* gpu_nums;
  cudaMalloc((void**) &gpu_nums, TESTE * sizeof(unsigned char));

//  curandState *states;
 // cudaMalloc((void**) &states, TESTE * sizeof(curandState_t));

  for(int i = 0; i < TESTE; i++)
  {
    cpu_nums[i] = 0;
    printf("%d - ", cpu_nums[i]);
  }
  persistentThreads<<<1, TESTE>>>(NULL, 0, 0, time(NULL), gpu_nums);

  cudaError_t err;
  err = cudaMemcpy(cpu_nums, gpu_nums, TESTE * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  printf("\n%u\n", err);
  for(int i = 0; i < TESTE; i++)
  {
    printf("%u - ", cpu_nums[i]);
  }


return 0;
}

*/


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
        /*
        //Init variables
        Individual *population, *nextPopulation;
        cudaMalloc((void**) &population, PSIZE * sizeof(Individual));
        cudaMalloc((void**) &nextPopulation, PSIZE * sizeof(Individual));

        Individual *cpu_population, *cpu_nextPopulation;
        cpu_population = (Individual *) malloc(PSIZE * sizeof(Individual));
        cpu_nextPopulation = (Individual *) malloc(PSIZE * sizeof(Individual));

*/

        float *maxFitness, *cpu_maxFitness;
        cudaMalloc((void**) &maxFitness, NGEN * sizeof(float));
        cpu_maxFitness = (float *) malloc(NGEN * sizeof(float));






        //persistentThreads<<<1, min(PSIZE, 1024), PSIZE * sizeof(Individual)>>>(PSIZE, NGEN, maxFitness, time(NULL));
 //       printf("%d\t%d", sizeof(Individual), PSIZE * sizeof(Individual));
        persistentThreads<<<1, min(PSIZE, 1024), PSIZE * sizeof(Individual)>>>(PSIZE, NGEN, maxFitness, time(NULL));

        //cudaMemcpy(cpu_population, population, PSIZE * sizeof(Individual), cudaMemcpyDeviceToHost);
        //cudaMemcpy(cpu_nextPopulation, nextPopulation, PSIZE * sizeof(Individual), cudaMemcpyDeviceToHost);
   //     printf("%s\n", cudaGetErrorString(cudaPeekAtLastError()));
        cudaMemcpy(cpu_maxFitness, maxFitness, NGEN * sizeof(float), cudaMemcpyDeviceToHost);
 //       printf("%s\n", cudaGetErrorString(cudaPeekAtLastError()));

/*
        for(int i = 0; i < PSIZE; i++)
        {
            printf("%f ; %x\n", cpu_population[i].fitness, cpu_population[i].chromossomes);
        }

        for(int i = 0; i < PSIZE; i++)
        {
            printf("%f ; %x\n", cpu_nextPopulation[i].fitness, cpu_nextPopulation[i].chromossomes);
        }
  */      
        end = clock();
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


        //free(cpu_population);

        //cudaFree(population);
        //cudaFree(nextPopulation);
    }

    printf("\nAvg T total(us)\t\tAvg T geração(us)\n");
    printf("%f\t\t%f\n", Ttotal/NIT, Ttotal/(NIT*NGEN));
    
return 0;
}

