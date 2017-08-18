#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

typedef struct {
  double *dset;
  unsigned int *mapping;
  int ldim;
  int sdim; 
} dinfo;

int max_iter = 100;

#define threshold 0.001


double compute_distance(double *v1, double *v2, int length){

  
  //printf("YOOOOOOOOOOOOoo%d %d %d\n", used, par_size, esize);
        
  int i = 0;
  double dist = 0;

  for(i=0; i<length; i++){
    dist += (v1[i] - v2[i])*(v1[i] - v2[i]); 
  }

  return(dist);
}


void process_batch(dinfo *din, dinfo *clusters, double *newCentroids, double* SumOfDist, double *sse){

  int i, j, k;
  double tmp_dist = 0;
  int tmp_index = 0;
  double min_dist = 0;
  double *dset = din->dset;
  double *centroids = clusters->dset;
  unsigned int *Index = din->mapping;
  unsigned int *cluster_size = clusters->mapping;

    
  for(i=0; i<clusters->sdim; i++){
    cluster_size[i] = 0;
  }

  for(i=0; i<din->sdim; i++){
    tmp_dist = 0;
    tmp_index = 0;
    min_dist = FLT_MAX;

    for(k=0; k<clusters->sdim; k++){
      tmp_dist = compute_distance(dset+i*din->ldim, centroids+k*clusters->ldim, din->ldim);
      if(tmp_dist<min_dist){
        min_dist = tmp_dist;
        tmp_index = k;
      }
    }
   
    Index[i] = tmp_index;
    SumOfDist[0] += min_dist;
  sse[0] += min_dist*min_dist;
    cluster_size[tmp_index]++;
    
  for(j=0; j<din->ldim; j++){
      newCentroids[tmp_index * clusters->ldim + j] += dset[i * din->ldim + j]; 
    }
   
  }

  for(k=0; k<clusters->sdim; k++){
    for(j=0; j<din->ldim; j++){
      centroids[k * clusters->ldim + j] = newCentroids[k * clusters->ldim + j];
    }
  }

}



void kmeans_cluster(dinfo *din, dinfo *clusters, int max_iter){ 

  int iter, i, j, k, dest;
  double SumOfDist = 0, new_SumOfDist = 0, part_SumOfDist, sse = 0, psse;
  double *newCentroids, *partCentroids;
  unsigned int *part_size;
  int endcond = 0;

  
  int rank, NumTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status stat;


  part_size = (unsigned int*) malloc(clusters->sdim*sizeof(unsigned int));
  newCentroids = (double*)malloc(clusters->ldim*clusters->sdim*sizeof(double));
  partCentroids = (double*)malloc(clusters->ldim*clusters->sdim*sizeof(double));


  for(iter=0; iter<max_iter; iter++){
            

    
    if(rank==0) {
  
        new_SumOfDist=0;

        for(k=0; k<clusters->sdim; k++) {
        part_size[k] = (unsigned int) 0;
        clusters->mapping[k] = 0;
      }

  
        MPI_Bcast(clusters->dset, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
      for(i=0; i<clusters->sdim; i++){
        for(j=0; j<clusters->ldim; j++){
          newCentroids[i * clusters->ldim + j] = 0;
        }
      }

        //partial sum
        MPI_Reduce(&part_SumOfDist, &new_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      //centroid sum
    MPI_Reduce(partCentroids,newCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
        //partial clusters' size
      for(dest = 1; dest<NumTasks; dest++) {
  
        MPI_Recv(part_size, clusters->sdim, MPI_UNSIGNED, dest, 5, MPI_COMM_WORLD, &stat);
  
        for(k=0; k<clusters->sdim; k++){
            clusters->mapping[k] += part_size[k];
        }
      }


         //  new center for each cluster
         for(k=0; k<clusters->sdim; k++) {
           for(j=0; j<clusters->ldim; j++) {
             clusters->dset[k * clusters->ldim + j] = newCentroids[k * clusters->ldim + j] / (double) clusters->mapping[k];
           }
         }
  
       
      if(fabs(SumOfDist - new_SumOfDist)<threshold){
        
          
          endcond = 1;
          MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
        break;

      } else {
      
        MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
      
      }
  
      SumOfDist = new_SumOfDist;
  
      printf("Squared Distances at iter %d: %f\n",iter, new_SumOfDist);

  } else {

        
    
        part_SumOfDist = 0;
        psse = 0;
  
        MPI_Bcast(clusters->dset, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
      for(i=0; i<clusters->sdim; i++){
        for(j=0; j<clusters->ldim; j++){
          partCentroids[i * clusters->ldim + j] = 0;
        }
      }
  
      process_batch(din, clusters, partCentroids, &part_SumOfDist, &psse);
  
      MPI_Reduce(&part_SumOfDist, &new_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      
      MPI_Reduce(partCentroids,newCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  
        
        MPI_Send(clusters->mapping, clusters->sdim, MPI_UNSIGNED, 0, 5, MPI_COMM_WORLD);


  
        MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
        if(endcond)
          break;

    }
  }


  if(rank==0) {
    
    MPI_Reduce(&psse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    printf("Finished after %d iterations\n", iter);
    printf("SSE equals to %f\n", sse);

  } else {

    MPI_Reduce(&psse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  free(newCentroids);
  free(partCentroids);
  free(part_size);
  
}


void scan_file(dinfo *din, FILE *fp){
  int i,j;
  int n = din->ldim;
  int m = din->sdim;
  double *tmp_dset = din->dset;
  double var;
  for(i = 0; i < m; i++)
  {
    for (j = 0 ; j < n; j++)
    {
      fscanf(fp,"%lf",&tmp_dset[i*n + j]);
      //printf("%d..%d..\n", i,j);
    }
    //printf("\n");
  }

  fclose(fp);
}

void synthetic_data(dinfo *din){

  int i, j = 0;
  int n = din->ldim;
  int m = din->sdim;
  double *tmp_dset = din->dset;
  unsigned int *tmp_Index = din->mapping;


  srand(0); 
  
  for(i=0; i<m; i++){
    tmp_Index[i] = 0;
    for(j=0; j<n; j++){
      tmp_dset[i*n + j] = (double) rand() / RAND_MAX; 
    }
  }

}


void init_clusters(dinfo *din,dinfo *cluster_in){

  int i, j, pick = 0;
  int n = cluster_in->ldim;
  int m = cluster_in->sdim;
  int Objects = din->sdim;
  double *tmp_Centroids = cluster_in->dset;
  double *tmp_dset = din->dset;
  unsigned int *tmp_Sizes = din->mapping;

  int step = Objects / m;

  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      tmp_Centroids[i*n + j] = tmp_dset[pick * n + j];
    }
    pick += step; 
  }

}

void print(dinfo* data2print){

  int i, j = 0;
  int n = data2print->ldim;
  int m = data2print->sdim;
  double *tmp_dset = data2print->dset;

  
  for(i=0; i<m; i++){
    for(j=0; j<n; j++){
      printf("%f ", tmp_dset[i*n + j]);
    }
    printf("\n");
  }
  
}


void save(dinfo* data2save, char *filename1, char *filename2){

  int i, j = 0;
  FILE *outfile;
  int n = data2save->ldim;
  int m = data2save->sdim;
  double *tmp_dset = data2save->dset;
  unsigned int *tmp_mapping = data2save->mapping;

  printf("Saving to files...\n");
  if((outfile=fopen(filename1, "wb")) == NULL){
    printf("Can't open \n");
  }

  for(i=0;i<m;i++){
    for(j=0;j<n;j++){
      fprintf(outfile, "%lf  ", tmp_dset[i*n+j]);
    }
    fprintf(outfile, "\n");
  }
  
  fclose(outfile);

  
  if((outfile=fopen(filename2, "wb")) == NULL){
    printf("Can't open \n");
  }

  for(i=0;i<m;i++){
    fprintf(outfile, "%d\n", tmp_mapping[i]);   
  }
  //fwrite(tmp_mapping, sizeof(unsigned int), m, outfile);

  fclose(outfile);

}

void clean(dinfo* data1){

  free(data1->dset);
  free(data1->mapping);
}

int main(int argc, char **argv){
	
	MPI_Init(&argc, &argv);

  struct timeval first, second, lapsed;
  struct timezone tzp;

  if(argc<4){
    return -1;
  }

  int numObjects = atoi(argv[1]);
  int numAttributes = atoi(argv[2]);
  int numClusters = atoi(argv[3]);
  int i =0 ;

  char *f1_0 = "centroids.txt";
  char *f1_1 = "Clusters.txt";
  char *f2_0 = "dset.txt";
  char *f2_1 = "mapping.txt"; 

  dinfo din;
  dinfo clusters;

  int rank, NumOfTasks;
  
  MPI_Comm_size(MPI_COMM_WORLD, &NumOfTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int NumOfSlaves = NumOfTasks-1;
  
  unsigned int parNumObjects;
  parNumObjects = numObjects/NumOfSlaves;
  
  int remain = numObjects - parNumObjects*NumOfSlaves;

  int procNumObjects[NumOfTasks];
  procNumObjects[0] = numObjects;

  for (i=1; i<NumOfTasks; i++) {

    if(i<=remain)
      procNumObjects[i] = parNumObjects+1;
    else
      procNumObjects[i] = parNumObjects; 
  }

  if (rank == 0) {

    din.ldim = numAttributes;
    din.sdim = numObjects;
    din.dset = (double*)malloc(numObjects*numAttributes*sizeof(double));
    din.mapping = (unsigned int*)malloc(numObjects*sizeof(unsigned int));

  } else {
    
    numObjects = procNumObjects[rank];

    din.ldim = numAttributes;
    din.sdim = numObjects;
    din.dset = (double*)malloc(numObjects*numAttributes*sizeof(double));
    din.mapping = (unsigned int*)malloc(numObjects*sizeof(unsigned int));

  }
  
  

  clusters.ldim = numAttributes;
  clusters.sdim = numClusters;
  clusters.dset = (double*)malloc(numClusters*numAttributes*sizeof(double));
  clusters.mapping = (unsigned int*)malloc(numClusters*sizeof(unsigned int)); 
  

  
  if (rank==0) {
    synthetic_data(&din);
    
    //FILE *fp = fopen("Iris_data", "r");
    //scan_file(&din, fp);

    init_clusters(&din, &clusters);
    printf("Data Scanning or Generation done\n");
  }

  

  int dest;
  double *sendBuf = (double *) malloc(numAttributes*procNumObjects[1]*sizeof(double));
  double *recvBuf = (double *) malloc(numAttributes*procNumObjects[rank]*sizeof(double));

  if (rank==0) {
  

    int offset = 0;
    for(dest=1; dest<NumOfTasks; dest++) {
     
      int cp;
      for(cp = 0; cp<procNumObjects[dest]*numAttributes; cp++) {
        sendBuf[cp] = din.dset[offset+cp];
      }
	  
      offset += procNumObjects[dest]*numAttributes;
      
	    MPI_Send(sendBuf, procNumObjects[dest]*numAttributes, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
    } 
  } else {

    MPI_Status stat;
    
    MPI_Recv(recvBuf, procNumObjects[rank]*numAttributes, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,&stat);

	int cp;
    
	for(cp=0; cp<procNumObjects[rank]*numAttributes; cp++) {
		din.dset[cp] = recvBuf[cp];
	}	
  }


  gettimeofday(&first, &tzp);

  kmeans_cluster(&din, &clusters, max_iter);


  if(rank==0) {
	
    MPI_Status stat;
    int offset = 0;
			
    unsigned int *rBuff = (unsigned int*) malloc(procNumObjects[1]*sizeof(unsigned int));

	for(dest=1; dest<NumOfTasks; dest++) {
	
		MPI_Recv(rBuff, procNumObjects[dest], MPI_UNSIGNED, dest, 10, MPI_COMM_WORLD, &stat);
				
		for(i=0; i<procNumObjects[dest]; i++) { 
			din.mapping[offset+i] = rBuff[i];
		}
					
		offset += procNumObjects[dest];
	}

	free(rBuff);

  } else {
  
	  MPI_Send(din.mapping, procNumObjects[rank], MPI_UNSIGNED, 0, 10, MPI_COMM_WORLD);		
  }

  gettimeofday(&second, &tzp);


  if(rank==0) {
  if(first.tv_usec>second.tv_usec){
    second.tv_usec += 1000000;
    second.tv_sec--;
  }
  
  lapsed.tv_usec = second.tv_usec - first.tv_usec;
  lapsed.tv_sec = second.tv_sec - first.tv_sec;

  printf("Time elapsed: %d.%06dsec\n", lapsed.tv_sec, lapsed.tv_usec); 

  
  save(&clusters, f1_0, f1_1);
  save(&din, f2_0, f2_1);

  }

  clean(&din);
  clean(&clusters);

  MPI_Finalize();
}


