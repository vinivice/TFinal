#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<algorithm>


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


int main(int argc, char *argv[ ]) 
{
    int PSIZE, NGEN, NIT;
    double Ttotal = 0;
    if(argc < 4)
    {
        printf("Uso %s <POP_SIZE> <N_GEN> <N_ITERACOES>\n", argv[0]);
        return 1;
    }
    else
    {
        PSIZE = atoi(argv[1]);
        NGEN = atoi(argv[2]);
        NIT = atoi(argv[3]);
    }

    printf("T total(us)\t\tT geração(us)\n");
    for(int it = 0; it < NIT; it++)
    {
        //Init variables
        Individual *population, *nextPopulation, *swap;
        population = (Individual *) malloc(PSIZE * sizeof(Individual));
        nextPopulation = (Individual *) malloc(PSIZE * sizeof(Individual));
        clock_t start, end;

        srandom(time(NULL));

        unsigned int mask = 0x3FF;
        ////printf("%x - %x - %x\n", mask, mask << 10, mask << 20);
        float a = 0, b = 0, c = 0;

        //Create population
        for(int i = 0; i < PSIZE; i++)
        {
            population[i].fitness = 0;
            population[i].chromossomes = random();
        }
    /*
        for(int i = 0; i < PSIZE; i++)
        {
            //printf("%f - ", population[i].fitness);
        }
    */
        start = clock();
        for(int i = 0; i < NGEN; i++)
        {
            //Calculate fitness
            float totalFitness = 0;
            for(int i = 0; i < PSIZE; i++)
            {
                a = population[i].chromossomes & mask;
                b = (population[i].chromossomes & (mask << 9)) >> 9;
                c = (population[i].chromossomes & (mask << 18)) >> 18;
         
                a = (a - 512)/100.0;
                b = (b - 512)/100.0;
                c = (c - 512)/100.0;

                population[i].fitness = 1.0 / (1 + a*a + b*b + c*c);
     //           //printf("%f - %f - %f - %f - %f\n", a, b, c, a*a + b*b + c*c, population[i].fitness);
                totalFitness += population[i].fitness;

            }
            std::sort(population, population + PSIZE, comparator);

            //printf("\n");
            /*for(int i = 0; i < PSIZE; i++)
            {
                //printf("%f;", population[i].fitness);
            }*/
            //printf("%f;%x", population[0].fitness, population[0].chromossomes);

            Individual parents[2];
            
            nextPopulation[0] = population[0];
            for(int i = 1; i < PSIZE; i++)
            {
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
                            parents[j] = population[k];
                            break;
                        }
                    }
                }
            
                //Crossover
                unsigned char cutPoint = random() % 28;
                unsigned mask1 = 0xffffffff << cutPoint; 
                unsigned mask2 = 0xffffffff >> (32 - cutPoint);
                Individual child;
                child.fitness = 0;
                child.chromossomes = (parents[0].chromossomes & mask1) + (parents[1].chromossomes & mask2);
      
                //Mutation
                float mutation = ((float) random() / RAND_MAX );
                if(mutation < MUT_PROB)
                {
                    unsigned char mutPoint = random() % 27;
                    child.chromossomes ^= 1 << mutPoint;
                }



                nextPopulation[i] = child;
            }
            swap = population;
            population = nextPopulation;
            nextPopulation = swap;




        }

        for(int i = 0; i < PSIZE; i++)
        {
            a = population[i].chromossomes & mask;
            b = (population[i].chromossomes & (mask << 9)) >> 9;
            c = (population[i].chromossomes & (mask << 18)) >> 18;
     
            a = (a - 512)/100.0;
            b = (b - 512)/100.0;
            c = (c - 512)/100.0;

            population[i].fitness = 1 / (1 + a*a + b*b + c*c);

        }
        std::sort(population, population + PSIZE, comparator);

        //printf("\n");
        
        /*for(int i = 0; i < PSIZE; i++)
        {
            //printf("%f;", population[i].fitness);
        }*/

        //printf("%f;%x", population[0].fitness, population[0].chromossomes);
        //printf("\n");
        
        end = clock();
        double cpu_time_used = 1000000 * ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("%f\t\t%f\n", cpu_time_used, cpu_time_used/NGEN);
        Ttotal += cpu_time_used;
    }

    printf("\nAvg T total(us)\t\tAvg T geração(us)\n");
    printf("%f\t\t%f\n", Ttotal/NIT, Ttotal/(NIT*NGEN));
    
 //   //printf("Hello World\n");

return 0;
}
