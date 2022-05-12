#include <iostream>
#include <vector>
#include <stdio.h>
#include <typeinfo>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <bits/stdc++.h>
#include <time.h> 



static const int blockSize = 1024;
static const int gridSize = 24; 

extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "stb_image.h"
}
                         
__global__ void reduce(const int *gArr, int pixels, int *gOut) {
    int thIdx = threadIdx.x;
    int gthIdx = thIdx + blockIdx.x*blockSize;
    const int gridSize = blockSize*gridDim.x;
    int sum = 0;
    for (int i = gthIdx; i < pixels; i += gridSize)
        sum += gArr[i];
    __shared__ int shArr[blockSize];
    shArr[thIdx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) { 
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        __syncthreads();
    }
    if (thIdx == 0)
        gOut[blockIdx.x] = shArr[0];
}

__host__ int avgArray(int* arr, int pixels) {
    //allocate device array and copy to GPU
    int* dev_arr;
    cudaMalloc((void**)&dev_arr, pixels * sizeof(int));
    cudaMemcpy(dev_arr, arr, pixels * sizeof(int), cudaMemcpyHostToDevice);
    
    //result allocation                        
    int sum;
    int* dev_out;
    cudaMalloc((void**)&dev_out, sizeof(int)*gridSize);
    
    //run kernel to reduce array                    
    reduce<<<gridSize, blockSize>>>(dev_arr, pixels, dev_out);
    //dev_out now holds the partial result
    reduce<<<1, blockSize>>>(dev_out, gridSize, dev_out);
    //dev_out[0] now holds the final result
    cudaDeviceSynchronize();
    
    //copy result back to host and free memory
    cudaMemcpy(&sum, dev_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dev_arr);
    cudaFree(dev_out);
    return sum/pixels;
}

class ImageInfo {
    public:
        int imgNo;
        int width;
        int height;
        int numPixels;                     
        int n;
        unsigned char *imageData;
        int *R;
        int *G;
        int *B;
        int *R_dev, *G_dev, *B_dev;
        int R_avg, G_avg, B_avg;
    
        ImageInfo(const char *fileName, int x){
            
            
            imgNo=x;
            imageData=stbi_load(fileName, &width, &height, &n, 0);
            numPixels=width*height;
            
           
            //HOST pinned memory allocation
            cudaMallocHost((void **)&R, width * height * sizeof(int));
            cudaMallocHost((void **)&G, width * height * sizeof(int));
            cudaMallocHost((void **)&B, width * height * sizeof(int));
            
            //Separate RGB Channels
            int y=0;
            
            for(int j=0; x< width * height; x++){
                R[j] = static_cast<int>(imageData[y+0]);
                G[j] = static_cast<int>(imageData[y+1]);
                B[j] = static_cast<int>(imageData[y+2]);
                y+=3;
            }
            
            
            
            
        }
    
        void launchKernel(){
            R_avg= avgArray(R, numPixels);
            G_avg= avgArray(G, numPixels);
            B_avg= avgArray(B, numPixels);
            
        }
        
        void freeArrays(){
            cudaFreeHost(R);
            cudaFreeHost(G);
            cudaFreeHost(B);
            delete [] imageData;
        }
    
};

//finds color distance in RGB space
double colorDistance(std::vector<ImageInfo>::iterator im1, std::vector<ImageInfo>::iterator im2){
    
    double radArg=  pow((im1->R_avg-im2->R_avg),2)
                   +pow((im1->G_avg-im2->G_avg),2)
                   +pow((im1->B_avg-im2->B_avg),2);
    
    return pow(radArg,.5);
    
}

//collects color distances of multiple images into array
double** colorDistanceGrid(int numFiles, std::vector<ImageInfo>::iterator begin){
    //output 2D array of doubles
    double** colorDistances;
    colorDistances = new double*[numFiles];
    
    //iterators
    std::vector<ImageInfo>::iterator im1=begin;
    std::vector<ImageInfo>::iterator im2=begin;
    
    
    for(int x=0; x<numFiles; x++){
        colorDistances[x]=new double[numFiles];                   
        im1=begin+x;
        for(int y=0; y<numFiles; y++){
            im2=begin+y;
            //color distance to self is 0
            if(x==y){
                colorDistances[x][y]=0;
            }
            //avoid double calculation
            if(x>y){
                colorDistances[x][y]=colorDistances[y][x];
            }
            //calc color distance
            else{
                colorDistances[x][y]=colorDistance(im1, im2);                
            }            
        }
    }    
    return colorDistances;  
}

//helper function prints 2darrays
void print2dArray(double **arr, int numFiles){   
    for(int x=0; x<numFiles; x++){        
        for(int y=0; y<numFiles; y++){
            std::cout << arr[x][y] << " ";
        }
        std::cout<< std::endl;
    }  
}

//helper function prints arrays
void printArray(int *arr, int numFiles, int gridX){   
    for(int x=0; x<numFiles; x++){        
        if(x%gridX==0){
        std::cout<< std::endl;
        }
        std::cout<< arr[x] << " ";
    }
    std::cout<< std::endl;
}
void printGrid(int *arr, char *args[], int numFiles, int gridX){
    for(int x=0; x<numFiles; x++){        
        if(x%gridX==0){
        std::cout<< std::endl;
        }
        std::cout<< "<" << args[arr[x]+2] << ">" << " ";
    }
    std::cout<< std::endl;  
}

//returns square of cartesian distance
double gridDistanceSq(int x1, int x2, int y1, int y2){
    return pow(x2-x1,2)+pow(y2-y1,2);
}

//score is sum of color distance/(grid distance)^2 for all image pairs in an arrangement
double getScore(int numFiles, int *currentArrangement, double **colorDistances, int xGrid){
    double score;
    int im1XPosition, im2XPosition, im1YPosition, im2YPosition;
    for(int im1=0; im1<numFiles; im1++){
        for(int im2=im1+1; im2<numFiles; im2++){
            //translate 1d array to 2d positions
            im1XPosition=im1%xGrid;
            im2XPosition=im2%xGrid;
            im1YPosition=im1/xGrid;
            im2YPosition=im2/xGrid;
            score+= colorDistances[currentArrangement[im1]][currentArrangement[im2]]/gridDistanceSq(im1XPosition, im2XPosition, im1YPosition, im2YPosition);
            //std::cout<< "image " << im1 << " and " << im2 << ":" << std::endl;
            //std::cout<< "color distance: " << colorDistances[im1][im2]  << std::endl; 
            //std::cout<< "grid distance: " << gridDistanceSq(im1XPosition, im2XPosition, im1YPosition, im2YPosition)  << std::endl;
            //std::cout<< "score contribution: " <<  colorDistances[im1][im2]/gridDistanceSq(im1XPosition, im2XPosition, im1YPosition, im2YPosition) << std::endl;                                                                                                                   
        }
    
    }
    return score;
}

//goes through permutations of array    
int* findBestArrangement(int numFiles, double **colorDistances, int xGrid){
       
    //create integer array to simulate file position permutations in grid
    int *grid=new int[numFiles];
    for(int x=0; x<numFiles; x++){
        grid[x]=x;
    }
    int n = sizeof(grid) / sizeof(grid[0]);
    std::sort(grid, grid+n);
    
    //store best arrangement and score
    int *bestArrangement=new int[numFiles];
    double curScore= getScore(numFiles, grid, colorDistances, xGrid);
    double bestScore=curScore;    
    
    do {        
        curScore=getScore(numFiles, grid, colorDistances, xGrid);       
        if(curScore<bestScore){                               
            bestScore=curScore;
            for(int x=0; x<numFiles; x++){
                bestArrangement[x]=grid[x];
            }
        }
    }while(std::next_permutation(grid, grid+numFiles));   
    
    return bestArrangement;   
}

int main(int argc, char *argv[])
{   
    int numFiles= argc-2;
    int xGrid = atoi(argv[1]);
    
    std::vector<ImageInfo> Images;
    
    //opening images
    clock_t start= clock();
    for(int x=2; x<argc; x++){
        Images.push_back(ImageInfo(argv[x], x-2));
    }
    //time to load images                     
    double imgLoad = (double) (clock()-start) / CLOCKS_PER_SEC;                     
                         
    std::vector<ImageInfo>::iterator it= Images.begin();
    
    //gpu kernel launches
    start = clock();
    for(it; it != Images.end(); it++)
    {
        it->launchKernel();
        it->freeArrays();
    }
    //time to run kernels
    double kernelRuns = (double) (clock()-start) / CLOCKS_PER_SEC;   
    
    it= Images.begin();
     
    //find best arrangement
    start = clock();
    double** colorDist= colorDistanceGrid(numFiles, it);
    int *bestArrangement = new int[numFiles];
    bestArrangement=findBestArrangement(numFiles, colorDist, xGrid);
    
    printArray(bestArrangement, numFiles, xGrid);
    
    printGrid(bestArrangement, argv, numFiles, xGrid);
    
    //time to find arrangement
    double gridTime = (double) (clock()-start) / CLOCKS_PER_SEC; 
    
    //timing metrics
    std::cout<<"Number of Images: " << numFiles << std::endl;
    std::cout<<"Image Size: " << Images.begin()->numPixels << " pixels" <<std::endl;
    std::cout<<"Image Load Time: " << imgLoad << "s" <<std::endl;
    std::cout<<"Kernel Run Time: " << kernelRuns << "s" <<std::endl;
    std::cout<<"Grid Run Time: " << gridTime << "s" <<std::endl;

    return 0;
}
