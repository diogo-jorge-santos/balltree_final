#include <math.h>

#include <stdlib.h>

#include <mpi.h>

#include <omp.h>

#include "gen_points.c"

#define WORLD MPI_COMM_WORLD
int n_dim;
int nprocs;
long count = 0;

void swap(double** a, double** b)
{
    double* temp = *a;
    *a = *b;
    *b = temp;
}

long partition (double ** arr, long low, long high)
{
    double* pivot = arr[high]; // pivot
    long i = (low - 1); // Index of smaller element and indicates the right position of pivot found so far

    for (long j = low; j <= high - 1; j++)
    {
        // If current element is smaller than the pivot
        if (arr[j][0] < pivot[0])
        {
            i++; // increment index of smaller element
            swap(&arr[i],& arr[j]);
        }
    }
    swap(&arr[i + 1],& arr[high]);
    return (i + 1);
}

void my_qsort(double** arr, long low, long high,int num_threads)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
        at right place */
        long pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        if(num_threads>1){
#pragma omp taskgroup
{
            
#pragma omp task
            my_qsort(arr, low, pi - 1,num_threads-num_threads/2);
#pragma omp task 
            my_qsort(arr, pi + 1, high,num_threads-num_threads/2);
            
}
        
        }
        else{

            my_qsort(arr, low, pi - 1,1);
            my_qsort(arr, pi + 1, high,1);
        }
        
        
    }
}


struct tree
{
    double *center;
    double rad;
    long id;
    struct tree *L;
    struct tree *R;
};

double eucl(double *aux1, double *aux2)
{
    //returns s
    double ret = 0.0;
    double aux = 0.0;

    for (int i = 0; i < n_dim; i++)
    {
        aux = aux1[i] - aux2[i];
        ret += aux * aux;
    }
    return ret;
}

double inner(double *a, double *b)
{
    double ret = 0.0;
    for (int i = 0; i < n_dim; i++)
    {
        ret += a[i] * b[i];
    }
    return ret;
}

double *median_center(double **orth_aux, long size, int num_threads)
{

    double *_orth = (double *)malloc(n_dim * size * sizeof(double));
    double **orth = (double **)malloc(size * sizeof(double *));

    for (long i = 0; i < size; i++)
    {
        orth[i] = &_orth[i * n_dim];
        for (int j = 0; j < n_dim; j++)
        {
            orth[i][j] = orth_aux[i][j];
        }
    }

    my_qsort(orth, 0, size-1,num_threads);

    double *ret = (double *)malloc(n_dim * sizeof(double));

    if (size % 2 == 0)
    {

        for (int i = 0; i < n_dim; i++)
        {
            ret[i] = (orth[size / 2][i] + orth[size / 2 - 1][i]) / 2;
        }
    }
    else
    {
        for (int i = 0; i < n_dim; i++)
        {
            ret[i] = orth[size / 2][i];
        }
    }
    free(_orth);
    free(orth);
    return ret;
}

double rad(double **data, double *center, long data_size)
{
    double ret = 0;
    for (long i = 0; i < data_size; i++)
    {
        double aux = eucl(data[i], center);
        if (aux > ret)
        {
            ret = aux;
        }
    }
    return sqrt(ret);
}

void fit(struct tree *node, double **dataset, long size, long id,int threads,int distributed)
{
    node->id = id;
    #pragma omp atomic
    count++;

    if (size <= 1)
    {
        node->center = dataset[0];
        free(dataset);
        node->rad = 0.0;
        node->L = (struct tree *)malloc(sizeof(struct tree));
        node->R = (struct tree *)malloc(sizeof(struct tree));
        node->L->id = -1;
        node->R->id = -1;
    }
    else
    {

        double aux_dist = 0;
        long a = 0;
        long b = 0;
        double temp_dist;
        for (long i = 1; i < size; i++)
        {
            temp_dist = eucl(dataset[0], dataset[i]);
            if (temp_dist > aux_dist)
            {
                aux_dist = temp_dist;
                a = i;
            }
        }

        aux_dist = 0;
        for (long j = 0; j < size; j++)
        {

            temp_dist = eucl(dataset[a], dataset[j]);
            if (temp_dist > aux_dist)
            {
                aux_dist = temp_dist;
                b = j;
            }
        }
        double *_orth_aux = (double *)malloc(n_dim * size * sizeof(double));
        double **orth_aux = (double **)malloc(size * sizeof(double *));

        double b_a[n_dim];

        double sub_aux[n_dim];

        for (int j = 0; j < n_dim; j++)
        {
            b_a[j] = dataset[b][j] - dataset[a][j];
        }

        double inner_b_a = inner(b_a, b_a);

        double aux;

        for (long i = 0; i < size; i++)
        {
            orth_aux[i] = &_orth_aux[i * n_dim];

            //(p-a)
            for (int j = 0; j < n_dim; j++)
            {
                sub_aux[j] = dataset[i][j] - dataset[a][j];
            }

            aux = inner(sub_aux, b_a) / inner_b_a;

            // (p-a).(b-a)/(b-a).(b-a) *(b-a) + a
            for (int j = 0; j < n_dim; j++)
            {
                orth_aux[i][j] = aux * b_a[j] + dataset[a][j];
            }
        }

        node->center = median_center(orth_aux, size,threads);

        node->rad = rad(dataset, node->center, size);

        size_t size1 = size / 2;

        double **ret1 = (double **)malloc(size1 * sizeof(double *));
        double *_ret1 = (double *)malloc(n_dim * size1 * sizeof(double *));
        for (long i = 0; i < size1; i++)
            ret1[i] = &_ret1[i * n_dim];

        if (size % 2 != 0)
        {
            size1 = size1 + 1;
        }

        double **ret2 = (double **)malloc(size1 * sizeof(double *));
        double *_ret2 = (double *)malloc(n_dim * size1 * sizeof(double *));
        for (long i = 0; i < size1; i++)
            ret2[i] = &_ret2[i * n_dim];

        long aux1 = 0;
        long aux2 = 0;
        for (long i = 0; i < size; i++)
        {
            if (orth_aux[i][0] < node->center[0])
            {
                for (int j = 0; j < n_dim; j++)
                {
                    ret1[aux1][j] = (dataset[i][j]);
                }
                aux1++;
            }

            else
            {
                for (int j = 0; j < n_dim; j++)
                {
                    ret2[aux2][j] = (dataset[i][j]);
                }
                aux2++;
            }
        }
        free(dataset[0]);
        free(dataset);
        free(_orth_aux);
        free(orth_aux);

        node->L = (struct tree *)malloc(sizeof(struct tree));
        node->R = (struct tree *)malloc(sizeof(struct tree));
        
        if(distributed==0){
            fit(node->L, ret1, aux1, 2 * id + 1,1,0);
            fit(node->R, ret2, aux2, 2 * id + 2,1,0);
        }
        else{
        if (id <= nprocs - 2 )
        {
            node->R->id = 2 * id + 2;
            //to the tranverse function not print this node
            node->R->rad = (double)-1;
            //aux
            MPI_Send(&aux2, 1, MPI_LONG, (id + 1), 1, WORLD);
            //ret
            MPI_Send(&(ret2[0][0]), aux2 * n_dim, MPI_DOUBLE, (id + 1), 2, WORLD);
            free(ret2[0]);
            free(ret2);
            fit(node->L, ret1, aux1, 2 * id + 1,threads,1);
        }
        else if(threads>1){
            
#pragma omp task
    fit(node->L,ret1,aux1,2*id+1,threads/2,1);
#pragma omp task
    fit(node->R,ret2,aux2,2*id+2,threads-threads/2,1);
            
        }
        else
        {

            fit(node->L, ret1, aux1, 2 * id + 1,1,1);
            fit(node->R, ret2, aux2, 2 * id + 2,1,1);
        }
    }
   }
}

void visit(struct tree *node)
{

    printf("%ld %ld %ld %lf ", node->id, node->L->id, node->R->id, node->rad);

    for (int i = 0; i < n_dim - 1; i++)
    {
        printf("%lf ", node->center[i]);
    }

    printf("%lf\n", node->center[n_dim - 1]);
    fflush(stdout);
}

void traverse(struct tree *node)
{

    if (node->rad == -1)
    {
        return;
    }
    if (node->rad == 0.0)
    {
        visit(node);
    }
    else
    {
        traverse(node->L);

        visit(node);

        traverse(node->R);
    }
}

int main(int argc, char *argv[])
{

    int me;
    double exec_time;
    long global_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    struct tree *aux = (struct tree *)malloc(sizeof(struct tree));
    int allThreads = omp_get_max_threads();
    n_dim = atoi(argv[1]);
    if(atol(argv[2])<8*nprocs){
        if (me == 0)
        {
        long np = atol(argv[2]);

        exec_time = -MPI_Wtime();
        double **data = get_points(argc, argv, &n_dim, &np);

        fit(aux, data, np, 0,1,0);
            
            
            
        exec_time += MPI_Wtime();
        fprintf(stderr, "%.1lf\n", exec_time);
        printf("%d %ld \n", n_dim, count);
        fflush(stdout);
        traverse(aux);

             

        }
    }
    else{
        
    MPI_Bcast(&n_dim, 1, MPI_LONG, 0, WORLD);

    if (me == 0)
    {
        long np = atol(argv[2]);

        exec_time = -MPI_Wtime();
        double **data = get_points(argc, argv, &n_dim, &np);
    #pragma omp parallel
  #pragma omp single
        fit(aux, data, np, 0,allThreads,1);
             #pragma omp taskwait

    }
    else
    {
        long np;
        MPI_Status status[2];
        MPI_Recv(&np, 1, MPI_LONG, MPI_ANY_SOURCE, 1, WORLD, &status[0]);
        double *_data = (double *)malloc(n_dim * np * sizeof(double));
        double **data = (double **)malloc(np * sizeof(double *));
        for (long i = 0; i < np; i++)
            data[i] = &_data[i * n_dim];

        MPI_Recv(&(data[0][0]), np * n_dim, MPI_DOUBLE, MPI_ANY_SOURCE, 2, WORLD, &status[1]);
        #pragma omp parallel
  #pragma omp single
        fit(aux, data, np, (long)me * 2,allThreads,1);
             #pragma omp taskwait

    }

    MPI_Barrier(WORLD);
    global_count = 0;
    MPI_Reduce(&count, &global_count, 1, MPI_LONG, MPI_SUM, 0, WORLD);

    if (me == 0)
    {
        exec_time += MPI_Wtime();
        fprintf(stderr, "%.1lf\n", exec_time);
        printf("%d %ld \n", n_dim, global_count);
        fflush(stdout);
    }
    MPI_Barrier(WORLD);

    int rank = 0;
    while (rank < nprocs)
    {
        if (me == rank)
        {
            traverse(aux);
            fflush(stdout);
        }
        rank++;
        MPI_Barrier(WORLD);
    }
    }
    MPI_Finalize();
    return 0;
}
