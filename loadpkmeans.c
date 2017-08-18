#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

int max_iter =50;
#define th 0.001

typedef struct {
  double *dset;
  unsigned int *mapping;
  int ldim;
  int sdim; 
} dinfo;

double compute_distance(double *v1, double *v2, int length){

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

  /*update cluster centers*/
  for(k=0; k<clusters->sdim; k++){
    for(j=0; j<din->ldim; j++){
      centroids[k * clusters->ldim + j] = newCentroids[k * clusters->ldim + j];
    }
  }

}


void cluster_first(dinfo *din, dinfo *clusters, int max_iter, double *extra, int esize, int procsize, int threshold){ 

  int iter, i, j, k, dest;
  double SumOfDist = 0, new_SumOfDist = 0, part_SumOfDist, sse = 0, psse;
  double *newCentroids, *partCentroids;
  unsigned int *part_size;
  int endcond = 0;

  int used = 0;
  int rank, NumTasks;
  MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status stat;
  int par_size;
  if(esize > procsize)
    par_size = esize/procsize;
  else
    par_size = procsize;

  part_size = (unsigned int*) malloc(clusters->sdim*sizeof(unsigned int));
  newCentroids = (double*)malloc(clusters->ldim*clusters->sdim*sizeof(double));
  partCentroids = (double*)malloc(clusters->ldim*clusters->sdim*sizeof(double));

  dinfo data_newin;
  data_newin.ldim = din->ldim;
  data_newin.sdim = 0;
  double *recvBuf = (double *)malloc(4*(threshold +1)*din->ldim*sizeof(double));
  data_newin.mapping = (unsigned int *)malloc(4*(threshold+1)*sizeof(unsigned int));
  data_newin.dset = recvBuf;


  MPI_Request req, req_1;
  MPI_Status status, status_1;
  int firstDone = 0;

  if(rank == 0){
    new_SumOfDist=0;
  
        for(k=0; k<clusters->sdim; k++) {
      part_size[k] = (unsigned int) 0;
      clusters->mapping[k] = 0;
    }

     for(i=0; i<clusters->sdim; i++){
        for(j=0; j<clusters->ldim; j++){
          newCentroids[i * clusters->ldim + j] = 0;
        }
      }

    MPI_Bcast(clusters->dset, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
    //printf("Broadcast successful.........\n");
      
  } else {
        used = -2;
        part_SumOfDist = 0;
        psse = 0;
  
      if(firstDone == 0)
            MPI_Bcast(clusters->dset, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  
      
  }

  while(used < esize) {         
    if(used == -99){
      break;
    }

    if(rank==0) {
  
      
       

      int flag = 0;    
      status.MPI_TAG = -1;
      ////printf("%d...\n", clusters->ldim*clusters->sdim);
      //MPI_Irecv(partCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &req);        
      //MPI_Test(&req, &flag, &status);
      MPI_Recv(partCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_ANY_SOURCE, 4, MPI_COMM_WORLD, &status);        
      flag = 1;
      if(flag != 0){
        //printf("YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOoo%d %d %d\n", used, par_size, esize);
        if(status.MPI_TAG == 4){
            for(k=0; k<clusters->sdim; k++){
              for(j=0; j<clusters->ldim; j++){
              newCentroids[k * clusters->ldim + j] += partCentroids[k * clusters->ldim + j];
              }
            }
            //printf("Waiting for process %d\n", status.MPI_SOURCE);  
            MPI_Recv(part_size, clusters->sdim, MPI_UNSIGNED, status.MPI_SOURCE, 5, MPI_COMM_WORLD, &stat);
            
            for(k=0; k<clusters->sdim; k++){
              clusters->mapping[k] += part_size[k];
            }
            //
            if (used + par_size <= esize){
              MPI_Send(extra + (used*din->ldim), din->ldim*par_size, MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD);    
              used += par_size;
              //printf("used......%d---------%d\n", used, par_size);
            } else {
              MPI_Send(extra + (used*din->ldim), din->ldim*(esize-used), MPI_DOUBLE, status.MPI_SOURCE, 1, MPI_COMM_WORLD);      
              used = esize;
            }
            //MPI_Send(extra, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            //used update
          
        }
      }

      //MPI_Cancel(&req);
      //MPI_Request_free(&req);
    
  } else {

        for(i=0; i<clusters->sdim; i++){
        for(j=0; j<clusters->ldim; j++){
          partCentroids[i * clusters->ldim + j] = 0;
        }
      }
  
        if(firstDone == 0)
          process_batch(din, clusters, partCentroids, &part_SumOfDist, &psse);
        else
          process_batch(&data_newin, clusters, partCentroids, &part_SumOfDist, &psse);    

        if(firstDone != 0)
          //printf("KMEANS PROCESSING DONE FOR PROCESS %d\n", rank);
      
        MPI_Send(partCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD);

        //sleep(1);
        //MPI_Reduce(partCentroids,newCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);  
        
        MPI_Send(clusters->mapping, clusters->sdim, MPI_UNSIGNED, 0, 5, MPI_COMM_WORLD);

         ////printf("done till here\n"); 
    
        MPI_Recv(recvBuf, 4*(threshold+1)*din->ldim, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,&stat);
        //printf("woaaaaaaaaaah\n");
        data_newin.dset = recvBuf;
        int recv_size;
        MPI_Get_count(&stat, MPI_DOUBLE, &recv_size);
        recv_size = recv_size/(din->ldim); 
        if(stat.MPI_TAG == 1){
          //printf("receive size is %d....\n", recv_size);
          data_newin.sdim = 0;
          data_newin.ldim = din->ldim;
          int size = din->sdim;
          int n = din->ldim;  
          for(i=0;i<recv_size;i++){
              for(j=0;j<din->ldim;j++){
                din->dset[size*n +j] = recvBuf[i*n+j];  
              }
              din->sdim++;
              data_newin.sdim++;               
          }
          firstDone = 1;
          //printf("first is doneeeeeeeeeeeeeeeeeeeeeeeee\n");
        } else if(stat.MPI_TAG == 99){
          //printf("game over for process %d\n", rank);
          used = -99;
          break;
        }
    }
  }

  if(rank == 0){
    int v = -1;
    // compute the new center for each cluster
    //printf("sending terminaion to all-----------------------------\n");
    for(dest =1; dest <= procsize; dest++){
      MPI_Send(&v, 1, MPI_DOUBLE, dest, 99, MPI_COMM_WORLD);      
    }

    for(k=0; k<clusters->sdim; k++) {
      for(j=0; j<clusters->ldim; j++) {
       clusters->dset[k * clusters->ldim + j] = newCentroids[k * clusters->ldim + j] / (double) clusters->mapping[k];
      }
    }
    

  }


  //printf("yipeeeeeeeeeeeeeeeeeeeeeee %d\n", rank);


  free(newCentroids);
  free(partCentroids);
  free(part_size);
  
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

        MPI_Reduce(&part_SumOfDist, &new_SumOfDist, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      

    
    MPI_Reduce(partCentroids,newCentroids, clusters->ldim*clusters->sdim, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      
      for(dest = 1; dest<NumTasks; dest++) {
  
        MPI_Recv(part_size, clusters->sdim, MPI_UNSIGNED, dest, 5, MPI_COMM_WORLD, &stat);
  
        for(k=0; k<clusters->sdim; k++){
            clusters->mapping[k] += part_size[k];
        }
      }


         for(k=0; k<clusters->sdim; k++) {
           for(j=0; j<clusters->ldim; j++) {
             clusters->dset[k * clusters->ldim + j] = newCentroids[k * clusters->ldim + j] / (double) clusters->mapping[k];
           }
         }
  
      if(fabs(SumOfDist - new_SumOfDist)<th){
        
          endcond = 1;
          MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
        break;

      } else {
      
        MPI_Bcast(&endcond, 1, MPI_INT, 0, MPI_COMM_WORLD);
      
      }
  
      SumOfDist = new_SumOfDist;
  
      //printf("Sum of Distances of iteration %d: %f\n",iter, new_SumOfDist);

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

    //printf("Finished after %d iterations\n", iter);
    //printf("SSE equals to %f\n", sse);

  } else {
    

    MPI_Reduce(&psse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  }

  free(newCentroids);
  free(partCentroids);
  free(part_size);
  
}





void random_initialization(dinfo *din){

  int i, j = 0;
  int n = din->ldim;
  int m = din->sdim;
  double *tmp_dset = din->dset;
  unsigned int *tmp_Index = din->mapping;


  
   srand(0); 
  // random floating points [0 1]
  for(i=0; i<m; i++){
    tmp_Index[i] = 0;
    for(j=0; j<n; j++){
      tmp_dset[i*n + j] = (double) rand() / RAND_MAX; 
    }
  }

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
      ////printf("%d..%d..\n", i,j);
    }
    ////printf("\n");
  }

  fclose(fp);
}


void initialize_clusters(dinfo *din,dinfo *cluster_in){

  int i, j, pick = 0;
  int n = cluster_in->ldim;
  int m = cluster_in->sdim;
  int Objects = din->sdim;
  double *tmp_Centroids = cluster_in->dset;
  double *tmp_dset = din->dset;
  unsigned int *tmp_Sizes = din->mapping;

  int step = Objects / m;

  /*randomly pick initial cluster centers*/
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
      //printf("%f ", tmp_dset[i*n + j]);
    }
    //printf("\n");
  }
  
}




void clean(dinfo* data1){

  free(data1->dset);
  free(data1->mapping);
}

int main(int argc, char **argv){

  int i,j,k;  
  // Pass arguments to MPI procceses  
  MPI_Init(&argc, &argv);

  struct timeval first, second, lapsed;
  struct timezone tzp;

  if(argc<4){
    return 0;
    ////printf("Error using kmeans: Three arguments required\n");
  }

  int numObjects = atoi(argv[1]);
  int numAttributes = atoi(argv[2]);
  int numClusters = atoi(argv[3]);
  i =0 ;

  dinfo din;
  dinfo clusters;
  
  int rank, NumOfTasks;
  
  MPI_Comm_size(MPI_COMM_WORLD, &NumOfTasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int NumOfSlaves = NumOfTasks-1;
  
  unsigned int parNumObjects;
  parNumObjects = numObjects/NumOfSlaves;
  int threshold = parNumObjects/2.5;
  int alert[NumOfSlaves+1];
  double splitters[NumOfTasks-1];

  // number of object that will not be included
  // if all processes get parNumObjects objects
  
  //int remain = numObjects - parNumObjects*NumOfSlaves;

  // compute final number of objects per task
  int procNumObjects[NumOfTasks];
  
  procNumObjects[0] = numObjects;

  for (i=1; i<NumOfTasks; i++) {
      procNumObjects[i] = 5*(threshold+1);
      alert[i] = 0; 
  }
  
  /*=======Memory Allocation=========*/

  // Allocate the appropriate memory
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

    din.sdim = 0;
  }

  ////printf("%d******************\n", numObjects);
    
  
 // //printf("Process %i will take %ld number of objects.\n", rank, numObjects);
  

  clusters.ldim = numAttributes;
  clusters.sdim = numClusters;
  clusters.dset = (double*)malloc(numClusters*numAttributes*sizeof(double));
  clusters.mapping = (unsigned int*)malloc(numClusters*sizeof(unsigned int)); 
  
  int n,m;
  /*=============Get dset==========*/

  if (rank==0) {
    //random_initialization(&din);
    FILE *fp = fopen("Iris_data", "r");
    scan_file(&din, fp);
    initialize_clusters(&din, &clusters);
    //printf("Data initiallized!\n");
    //save(&din, file2_0_0, file2_1_1);

    n = din.ldim;
    m = din.sdim;
    
    double min = FLT_MAX;
    double max = FLT_MIN;
    
    for (i=0;i<m;i++){
      if(max <din.dset[i*n] ){
        max = din.dset[i*n];
      }
      if(min > din.dset[i*n] ){
        min = din.dset[i*n];
      } 
    }

    ////printf("min is %lf and max is %lf\n", min, max);
    double psize = ((max-min))/(NumOfTasks-1);

    for(i=0;i<NumOfTasks-2;i++){
      splitters[i] = min + ((i+1)*psize);
      ////printf("%lf.....%d\n", splitters[i], i);
    }

  }
  

  /*=============Sending dset============*/


  // send dset to other process.
  // will only send the appropriate part
  // of the dset
  int dest;
  double *sendBuf = (double *) malloc(numAttributes*sizeof(double));
  double *recvBuf = (double *) malloc(numAttributes*sizeof(double));

  double *extra;
  int esize=-1;
  if (rank==0) {
  
  // holds the start position to copy elements to sendBuf
    /*
    int offset = 0;
    for(dest=1; dest<NumOfTasks; dest++) {
     
      int cp;
      for(cp = 0; cp<procNumObjects[dest]*numAttributes; cp++) {
        sendBuf[cp] = din.dset[offset+cp];
      }
    // update offset
      offset += procNumObjects[dest]*numAttributes;
      
    // send dset
      MPI_Send(sendBuf, procNumObjects[dest]*numAttributes, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
    }
    */
    //printf("Theshold is %d......\n", threshold);
    for (i=1; i<NumOfTasks; i++) {
      procNumObjects[i] = 0; 
    }
    extra = (double *) malloc(4*parNumObjects*numAttributes*sizeof(double));
    esize=0;
  }  
  
  ////printf("%d***************####\n", sizeof(din.dset));
  if(rank ==0){
    //printf("%d %d******************\n", m, n);
    for (i=0;i<m;i++){
        ////printf("Iteration %d----------------\n", i);
        for(j=1;j<NumOfTasks-1;j++){
          ////printf("%d----%lf", i, din.dset[i*n]);
          if(splitters[j-1] > din.dset[i*n]){
            dest = j;
            break;  
          }
        }  
        if(j== (NumOfTasks-1)){
          dest = j;    
        }
        ////printf("Destination is %d for %d\n", dest, i);
        int cp;        
        if(procNumObjects[dest] == threshold){
          ////printf("Storing extra object %d at size %d\n", i, esize);
          for(cp = 0; cp<numAttributes; cp++) {
            extra[esize*n+cp] = din.dset[i*n+cp];
          }
          alert[dest]=1;
          ////printf("ALERTTT.............. for %d\n", dest);
          esize++;
          continue;    
        }
        ////printf("%d&&&&&&&&&&&&&&\n", procNumObjects[dest]);  
        procNumObjects[dest]++;
        ////printf("Sending object %d to %d\n", i, dest);
        ////printf("%d&&&&&&&&&&&&&&\n", procNumObjects[dest]);  
        for(cp = 0; cp<numAttributes; cp++) {
          sendBuf[cp] = din.dset[i*n+cp];
        }
        MPI_Send(sendBuf, numAttributes, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
        ////printf("Esize is %d.....\n", esize);
      }
      ////printf("heloooooooooooooooooooooooooooooooooooo\n");
      for(dest=1;dest<NumOfTasks;dest++){
        ////printf("procNumObjects %d\n", procNumObjects[dest]);
        ////printf("sending dummy....to %d\n", dest);
        MPI_Send(&dest, 1, MPI_INT, dest, 101, MPI_COMM_WORLD);
      }
  } else {
      din.sdim = 0;
      MPI_Status stat;      
      while(1){
           // Receive dset
        MPI_Recv(recvBuf, numAttributes, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD,&stat);

        if(stat.MPI_TAG == 1){
          ////printf("Process %d Receiving from zero\n", rank);

          int cp;
          int size = din.sdim;
          int n = din.ldim;  
          ////printf("Process %d-%d %d %d\n", rank, size, n, sizeof(din.dset));
          // copy elements from recvB to din.dset array
          for(cp=0; cp<numAttributes; cp++) {
            din.dset[size*n +cp] = recvBuf[cp];
          }
          din.sdim++;          
        } else if(stat.MPI_TAG == 101){
          ////printf("Process %d  receving dummy\n", rank);
          break;
        }
      }
  }
  
  //printf("*******************************************Done till here..............\n");

  gettimeofday(&first, &tzp);

  cluster_first(&din, &clusters, max_iter, extra, esize, NumOfTasks-1, threshold);
  kmeans_cluster(&din, &clusters, max_iter);


  if(rank==0) {
  
    MPI_Status stat;
    int offset = 0;
    int offset_1 = 0; 
    unsigned int *rBuff = (unsigned int*) malloc(4*(threshold+1)*sizeof(unsigned int));
    unsigned int *rBuff_1 = (unsigned int*) malloc(4*(threshold+1)*(din.ldim)*sizeof(double));


  for(dest=1; dest<NumOfTasks; dest++) {
  
    MPI_Recv(rBuff_1, 4*(threshold+1)*din.ldim, MPI_DOUBLE, dest, 102, MPI_COMM_WORLD, &stat);
    int recv_size_1;
    MPI_Get_count(&stat, MPI_DOUBLE, &recv_size_1);
    recv_size_1 = recv_size_1/din.ldim;
    //printf("receiving size is %d from %d\n", recv_size_1, dest);
    for(i=0; i<recv_size_1; i++) { 
      for(j=0;j<din.ldim;j++){
        ////printf("ready...\n");
        din.dset[((offset_1+i)*din.ldim) + j] = rBuff_1[(i*din.ldim)+j];
      }
    }
          
    offset_1 += recv_size_1;
    


    // Receive indexes of elements
    MPI_Recv(rBuff, 4*(threshold+1), MPI_UNSIGNED, dest, 10, MPI_COMM_WORLD, &stat);
    int recv_size;
    MPI_Get_count(&stat, MPI_UNSIGNED, &recv_size);
    //printf("receiving size is %d from %d\n", recv_size, dest);
    
    for(i=0; i<recv_size; i++) { 
      ////printf("ready...\n");
      din.mapping[offset+i] = rBuff[i];
    }
          
    offset += recv_size;
    
  }

  free(rBuff);

  } else {
  
    // Send indexes of elements
    MPI_Send(din.dset, din.sdim*din.ldim, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD);
    MPI_Send(din.mapping, din.sdim, MPI_UNSIGNED, 0, 10, MPI_COMM_WORLD);    
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

  }

  clean(&din);
  clean(&clusters);

  MPI_Finalize();
}
