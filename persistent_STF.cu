#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>


//#define PSIZE 10
//#define NGEN 500000
#define MUT_PROB 0.5
#define CHROMO_SIZE 44

struct Individual 
{
    float fitness;
    char chromossomes[CHROMO_SIZE];
};

bool comparator (Individual i, Individual j)
{
    return (i.fitness > j.fitness);
}

void printPop(Individual *population, int popSize)
{
    printf("%f = \n", population[0].fitness);
    /*
    for(int i = 0; i < popSize; i++)
    {
        printf("%f = \n", population[i].fitness);
        for(int j = 0; j < CHROMO_SIZE; j++)
        {
                printf("%.2f;", (population[i].chromossomes[j] / 128.0) * 5.0);
        }
        printf("\n");
    }
    printf("=======================\n");
    */
}

float fitness(Individual *population, int popSize)
{
    float totalFitness = 0;
    for(int i = 0; i < popSize; i++)
    {
        float subTotal = 0;
        for(int j = 0; j < CHROMO_SIZE; j++)
        {
            float x = (population[i].chromossomes[j] / 128.0) * 5.0;
            subTotal += (x*x*x*x - 16.0*x*x + 5.0*x) / 2.0;
        }
        population[i].fitness = 1.0 / (40.0*CHROMO_SIZE + subTotal);
        totalFitness += population[i].fitness;
    }
    return totalFitness;
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
        //Init variables
        Individual *population, *nextPopulation, *swap;
        population = (Individual *) malloc(PSIZE * sizeof(Individual));
        nextPopulation = (Individual *) malloc(PSIZE * sizeof(Individual));
        clock_t start, end;

        float *maxFitness;
        maxFitness = (float *) malloc(NGEN * sizeof(float));

        srandom(time(NULL));

        float totalFitness = 0;


        //Create population
        for(int i = 0; i < PSIZE; i++)
        {
            population[i].fitness = 0;
            for(int j = 0; j < CHROMO_SIZE; j++)
            {
                population[i].chromossomes[j] = random()%256 - 128;
            }
        }
 //       printPop(population, PSIZE);

        start = clock();
        for(int i = 0; i < NGEN; i++)
        {
            //Calculate fitness
            totalFitness = fitness(population, PSIZE);
            std::sort(population, population + PSIZE, comparator);
            maxFitness[i] = population[0].fitness;

            //printf("\n");
            //printPop(population, PSIZE);
            //printf("%f;%x", population[0].fitness, population[0].chromossomes);

            Individual parents[2];
            int temp = -1;
            
            nextPopulation[0] = population[0];
            /*
            for(int j = 0; j < CHROMO_SIZE; j++)
            {
                population[1].chromossomes[j] = random()%256 - 128;
            }
            */
            for(int i = 1; i < PSIZE; i++)
            {
                //TODO Function reproduce
                //Selection
                for(int j = 0; j < 2; j++)
                {
                    float p = ((float) random() / RAND_MAX ) * totalFitness;
                    float score = 0;

                    for(int k = 0; k < PSIZE; k++)
                    {
                        score += population[k].fitness;
                        if(p < score)
                        {
                            if (k != temp)
                            {
                                parents[j] = population[k];
                                temp = k;
                            }
                            else
                            {
                                j--;
                            }
                            break;
                        }
                    }
                }
            
                //Crossover
                unsigned char cutPoint = random() % (CHROMO_SIZE + 1);
                Individual child;
                child.fitness = 0;
                for(int i = 0; i < cutPoint; i++)
                {
                    child.chromossomes[i] = parents[0].chromossomes[i];
                }

                for(int i = cutPoint; i < CHROMO_SIZE; i++)
                {
                    child.chromossomes[i] = parents[1].chromossomes[i];
                }

                //Mutation
                float mutation = ((float) random() / RAND_MAX );
                if(mutation < MUT_PROB)
                {
                    int mutPoint = random() % CHROMO_SIZE;
                    child.chromossomes[mutPoint] = random()%256 - 128;
                }



                nextPopulation[i] = child;
                //nextPopulation[i] = reproduce(PARAM);
                
            }
            swap = population;
            population = nextPopulation;
            nextPopulation = swap;
        }

        fitness(population, PSIZE);
        std::sort(population, population + PSIZE, comparator);

        //printf("\n");
        //printPop(population, PSIZE);
        
        //printf("%f;%x", population[0].fitness, population[0].chromossomes);
        //printf("\n");
        
        end = clock();
        free(population);
        free(nextPopulation);

        if(PRINT != 0)
        {
            printf("Gen\tFitness\n");
            for(int i = 0; i < NGEN; i++)
            {
                printf("%d\t%f\n", i+1, maxFitness[i]);
            }
        }

        printf("\nT total(us)\t\tT geração(us)\n");
        double cpu_time_used = 1000000 * ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("%f\t\t%f\n\n", cpu_time_used, cpu_time_used/NGEN);
        Ttotal += cpu_time_used;
    }

    printf("\nAvg T total(us)\t\tAvg T geração(us)\n");
    printf("%f\t\t%f\n", Ttotal/NIT, Ttotal/(NIT*NGEN));
    
return 0;
}
