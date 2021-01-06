#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <iostream>
#include <cstdlib>
#include <chrono> 
#include <algorithm>
#include <omp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <ppl.h>
#include <fstream>

using std::cin;
using std::cout;
using std::endl;
using std::string;


template<typename F>
double benchmark(F& lambda);
int32_t* generateArray(const const size_t& size);
int32_t* readArray(const string& path, const size_t& size);
void writeArray(const string& path, const size_t& size, int32_t* array);
bool isSorted(int32_t* array, const size_t& size);

void oddEvenSort(int32_t* array, const size_t& size);
void oddEvenSortParallel(int32_t* arr, const size_t& size, const int& threads);
void mergeSort(int32_t* array, const size_t& left, const size_t& right, int32_t* tempArray);
void mergeSortParallel(int32_t* array, const size_t& left, const size_t& right, const int& threads, int32_t* tempArray);
inline void mergeParallel(int32_t* array, const size_t& left, const size_t& center, const size_t& right, int32_t* tempArray);

void bitonicSortParallel(int32_t* array, const size_t& start, const size_t& size, bool dir, const int& threads);
void bitonicSort(int32_t* array, const size_t& start, const size_t& size, bool dir);
void bitonicMerge(int32_t* array, const size_t& start, const size_t& size, bool dir);
size_t lessPower2Than(const size_t& size);

__global__ void oddEvenSort_GPU(int32_t* arrayOnGPU, const unsigned int parity, const size_t size);


int main() {
    int choice;
    int numbersOfAlg;
    const unsigned int maxThreads = omp_get_max_threads();
    unsigned int currentThreads;
    int tmp;
    bool sortChoice[12] = { false };
    bool readArrayFromFile = false;
    string fileInName;
    string fileOutName;
    size_t size;
    do {
        for (int i = 0; i < 12; i++) {
            sortChoice[i] = false;
        }
        cout << "-----------------Menu-----------------" << endl;
        cout << "1 - Porowanie sortowan" << endl;
        cout << "2 - Sortowanie danych z pliku" << endl;
        cout << "0 - Koniec" << endl;
        cin >> choice;
        if (choice == 1) {
            cout << "----------Porowanie sortowan----------" << endl;
            cout << "Podaj ilosc liczb do sortowania:" << endl;
            cin >> size;
            cout << "-----------------CPU------------------" << endl;
            cout << "Dostepne watki: "<< maxThreads << endl;
            cout << "Podaj liczbe wykozystywanych watkow:  " << endl;
            cin >> currentThreads;
            cout << "--------------------------------------" << endl;
            cout << "0  - Odd-Even Sort" << endl;
            cout << "1  - Odd-Even Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "2  - Merge Sort" << endl;
            cout << "3  - Merge Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "4  - Bitonic Sort" << endl;
            cout << "5  - Bitonic Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "6  - qsort " << endl;
            cout << "7  - std::sort" << endl;
            cout << "----------------GPU-------------------" << endl;
            cout << "8  - Odd-Even Sort GPU" << endl;
            cout << "9  - Odd-Even Sort GPU with copy time" << endl;
            cout << "10 - trust::sort" << endl;
            cout << "11 - trust::sort with copy time" << endl;
            cout << "--------------------------------------" << endl;
            cout << "Wybierz ilosc algorytmow sortowania " << endl;
            cin >> numbersOfAlg;
            cout << "Podaj numery sortowan(oddzielone spacja) " << endl;
            for (int i = 0; i < numbersOfAlg; i++) {
                cin >> tmp;
                sortChoice[tmp] = true;
            }
            cout << "--------------------------------------" << endl <<endl;
        } else if (choice == 2) {
            cout << "-------Sortowanie liczb z pliku-------" << endl;
            cout << "Podaj nazwe pliku do odczytu:" << endl;
            cin >> fileInName;
            cout << "Podaj nazwe pliku do zapisu:" << endl;
            cin >> fileOutName;
            cout << "Podaj ilosc liczb do sortowania:" << endl;
            cin >> size;
            cout << "-----------------CPU------------------" << endl;
            cout << "Dostepne watki: " << maxThreads << endl;
            cout << "Podaj liczbe wykozystywanych watkow:  " << endl;
            cin >> currentThreads;
            cout << "--------------------------------------" << endl;
            cout << "0  - Odd-Even Sort" << endl;
            cout << "1  - Odd-Even Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "2  - Merge Sort" << endl;
            cout << "3  - Merge Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "4  - Bitonic Sort" << endl;
            cout << "5  - Bitonic Sort Parallel" << endl;
            cout << "--------------------------------------" << endl;
            cout << "6  - qsort " << endl;
            cout << "7  - std::sort" << endl;
            cout << "----------------GPU-------------------" << endl;
            cout << "8  - Odd-Even Sort GPU" << endl;
            cout << "9  - Odd-Even Sort GPU with copy time" << endl;
            cout << "10 - trust::sort" << endl;
            cout << "11 - trust::sort with copy time" << endl;
            cout << "--------------------------------------" << endl;
            cout << "Wybierz algorytm do sortowania " << endl;
            cin >> tmp;
            sortChoice[tmp] = true;
            readArrayFromFile = true;
            cout << "------------------------------------" << endl << endl;
        }
    
    int32_t* array = nullptr;;
    int32_t* arrayOnGPU = nullptr;
    int32_t* tempArray = new int32_t[size];

    omp_set_nested(1);
    omp_set_dynamic(0);

    cudaError_t cudaStatus;
    if ((cudaStatus = cudaSetDevice(0)) != cudaSuccess) {
        fprintf(stderr, "function: cudaSetDevice, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
        exit(-1);
    }

    //############# OddEvenSort #############  
    if (sortChoice[0]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double oddEvenSortTime = benchmark([&]() {
            oddEvenSort(array, size);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Odd Even Sort Time: " << oddEvenSortTime << endl;
    }
    //############# OddEvenSort Parallel #############
    if (sortChoice[1]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double oddEvenSortTimeParallel = benchmark([&]() {
            oddEvenSortParallel(array, size, currentThreads);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Odd Even Sort Parallel " << currentThreads << " threads Time: " << oddEvenSortTimeParallel << endl;
    }


    //############# MergeSort  #############
    if (sortChoice[2]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        } else {
            array = generateArray(size); 
        }
        double mergeSortTime;
        mergeSortTime = benchmark([&]() {
            mergeSort(array, 0, size - 1, tempArray);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Merge Sort Time: " << mergeSortTime << endl;
    }

    //############# MergeSort Parallel #############
    if (sortChoice[3]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double mergeSortParallelTime = benchmark([&]() {
            mergeSortParallel(array, 0, size - 1, currentThreads, tempArray);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Merge Sort Paralel "<< currentThreads <<" threads Time: " << mergeSortParallelTime << endl;
    }
  

    //############# Bitonic Sort #############
    if (sortChoice[4]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double bitonicSortTime = benchmark([&]() {
            bitonicSort(array, 0, size, 1);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Bitonic Sort Time: " << bitonicSortTime << endl;
    }
    //############# Bitonic Sort Parallel #############  
    if (sortChoice[5]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double bitonicSortParallelTime = benchmark([&]() {
            bitonicSortParallel(array, 0, size, 1, currentThreads);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Bitonic Sort Parallel " << currentThreads <<" threads Time: " << bitonicSortParallelTime << endl;
    }

    //############# qsort  #############
    if (sortChoice[6]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double qSortTime = benchmark([&]() {
            qsort(array, size, sizeof(int32_t), [](const void* a, const void* b) {
                return *static_cast<const int32_t*>(a) - *static_cast<const int32_t*>(b);
                });
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "qsort Time: " << qSortTime << endl;
    }
    //############# std::sort  #############
    if (sortChoice[7]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double stdSortTime = benchmark([&]() {
            std::sort(array, array + size);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "std::sort Time: " << stdSortTime << endl;
    }

    //############# OddEvenSort GPU #############
    if (sortChoice[8]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        if ((cudaStatus = cudaMalloc(reinterpret_cast<void**>(&arrayOnGPU), size * sizeof(int32_t))) != cudaSuccess) {
            fprintf(stderr, "function: cudaMalloc, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        if ((cudaStatus = cudaMemcpy(arrayOnGPU, array, size * sizeof(int32_t), cudaMemcpyHostToDevice)) != cudaSuccess) {
            fprintf(stderr, "function: cudaMemcpy, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        double oddEvenSortGPUTime = benchmark([&]() {

            for (size_t i = 0; i < size; i++) {
                oddEvenSort_GPU << < (size / 2 + 1023) / 1024, 1024 >> > (arrayOnGPU, i % 2, size);
            }

            });
        if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
            fprintf(stderr, "function: cudaGetLastError, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
            fprintf(stderr, "function: cudaDeviceSynchronize, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }
        if ((cudaStatus = cudaMemcpy(array, arrayOnGPU, size * sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
            fprintf(stderr, "function: cudaMemcpy, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        if ((cudaStatus = cudaFree(arrayOnGPU)) != cudaSuccess) {
            fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(-1);
        }

        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Odd Even Sort Time GPU without copy: " << oddEvenSortGPUTime << endl;
    }
    if (sortChoice[9]) {
        //#############  OddEvenSort GPU (copy) #############
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double oddEvenSortGPUCopyTime = benchmark([&]() {

            if ((cudaStatus = cudaMalloc(reinterpret_cast<void**>(&arrayOnGPU), size * sizeof(int32_t))) != cudaSuccess) {
                fprintf(stderr, "function: cudaMalloc, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            if ((cudaStatus = cudaMemcpy(arrayOnGPU, array, size * sizeof(int32_t), cudaMemcpyHostToDevice)) != cudaSuccess) {
                fprintf(stderr, "function: cudaMemcpy, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            for (size_t i = 0; i < size; i++) {
                oddEvenSort_GPU << < (size / 2 + 1023) / 1024, 1024 >> > (arrayOnGPU, i % 2, size);
            }

            if ((cudaStatus = cudaGetLastError()) != cudaSuccess) {
                fprintf(stderr, "function: cudaGetLastError, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) {
                fprintf(stderr, "function: cudaDeviceSynchronize, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            if ((cudaStatus = cudaMemcpy(array, arrayOnGPU, size * sizeof(int32_t), cudaMemcpyDeviceToHost)) != cudaSuccess) {
                fprintf(stderr, "function: cudaMemcpy, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            if ((cudaStatus = cudaFree(arrayOnGPU)) != cudaSuccess) {
                fprintf(stderr, "function: cudaFree, code: %d msg: %s", cudaStatus, cudaGetErrorString(cudaStatus));
                exit(-1);
            }

            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Odd Even Sort Time GPU with copy: " << oddEvenSortGPUCopyTime << endl;
    }
    //#############  thrust::sort #############
    if (sortChoice[10]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        thrust::device_vector<int32_t> vecDevice(size);
        copy(array, array + size, vecDevice.begin());
        double thrustSortTime = benchmark([&]() {
            thrust::sort(vecDevice.begin(), vecDevice.end(), thrust::less<int32_t>());
            });
        copy(vecDevice.begin(), vecDevice.end(), array);
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Thrust Sort Time: " << thrustSortTime << endl;
    }


        //#############  thrust::sort(coppy) #############
    if (sortChoice[11]) {
        if (readArrayFromFile) {
            array = readArray(fileInName, size);
        }
        else {
            array = generateArray(size);
        }
        double thrustSortTimeCopy = benchmark([&]() {
            thrust::device_vector<int32_t> vecDevice(size);
            copy(array, array + size, vecDevice.begin());
            thrust::sort(vecDevice.begin(), vecDevice.end(), thrust::less<int32_t>());
            copy(vecDevice.begin(), vecDevice.end(), array);
            });
        if (!isSorted(array, size)) {
            cout << "Error" << endl;
        }
        if (readArrayFromFile) {
            writeArray(fileOutName, size, array);
        }
        delete[] array;
        cout << std::fixed << "Thrust Sort Time with copy: " << thrustSortTimeCopy << endl;
    }
    } while (choice != 0);
    return 0;
       
}

template <typename T>
__device__ void inline swap(T& x, T& y) {
    T z(x); x = y; y = z;
}
__global__ void oddEvenSort_GPU(int32_t* arrayOnGPU, const unsigned int parity, const size_t size) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (parity && index * 2 + 2 < size) {
        if (arrayOnGPU[index * 2 + 1] > arrayOnGPU[index * 2 + 2]) {
            swap(arrayOnGPU[index * 2 + 1], arrayOnGPU[index * 2 + 2]);
        }
    } else if (!parity && index * 2 + 1 < size) {
        if (arrayOnGPU[index * 2] > arrayOnGPU[index * 2 + 1]) {
            swap(arrayOnGPU[index * 2], arrayOnGPU[index * 2 + 1]);
        }
    }
}

//############# Odd-Even Sort Utils  #############
void oddEvenSort(int32_t* array, const size_t& size) {
    for (size_t j = 0; j < size; j++) {
        if (j & 1) {
            for (size_t i = 2; i < size; i += 2)
                if (array[i - 1] > array[i])
                    std::swap(array[i - 1], array[i]);
        }
        else {
            for (size_t i = 1; i < size; i += 2)
                if (array[i - 1] > array[i])
                    std::swap(array[i - 1], array[i]);
        }
    }
}

void oddEvenSortParallel(int32_t* array, const size_t& size, const int& threads) {
    long i, j;
    int chunk = size < 1000 ? 10 : 100;
#pragma omp parallel private(j) num_threads(threads)
    {
        for (j = 0; j < size; j++) {
            if (j & 1) {
#pragma omp for schedule(guided, chunk)
                for (i = 2; i < size; i += 2)
                    if (array[i - 1] > array[i])
                        std::swap(array[i - 1], array[i]);
            }
            else {
#pragma omp for schedule(guided, chunk)
                for (i = 1; i < size; i += 2)
                    if (array[i - 1] > array[i])
                        std::swap(array[i - 1], array[i]);
            }
        }
    }
}


    //############# MergeSort Utils  #############

inline void mergeParallel(int32_t* array, const size_t& left, const size_t& center, const size_t& right, int32_t* tempArray) {
    size_t i = left;
    size_t j = center + 1;
    size_t current = left;
    while (i <= center && j <= right) {
        if (array[j] < array[i]) {
            tempArray[current++] = array[j++];
        }
        else {
            tempArray[current++] = array[i++];
        }
    }
    if (i <= center) {
        while (i <= center) {
            tempArray[current++] = array[i++];
        }
    }
    else {
        while (j <= right) {
            tempArray[current++] = array[j++];
        }
    }
    memcpy(array + left, tempArray + left, (right - left + 1) * sizeof(int32_t));
}

void mergeSort(int32_t* array, const size_t& left, const size_t& right, int32_t* tempArray) {
    if (left < right) {
        const size_t center = (left + right) / 2;
        mergeSort(array, left, center, tempArray);
        mergeSort(array, center + 1, right, tempArray);
        mergeParallel(array, left, center, right, tempArray);
    }
}

void mergeSortParallel(int32_t* array, const size_t& left, const size_t& right, const int& threads, int32_t* tempArray) {
    if (threads == 1) {
        mergeSort(array, left, right, tempArray);
    }
    else if (threads > 1) {
        if (left < right) {
            const size_t center = (left + right) / 2;
            #pragma omp parallel sections num_threads(2)
            {
                #pragma omp section
                mergeSortParallel(array, left, center, threads / 2, tempArray);
                #pragma omp section
                mergeSortParallel(array, center + 1, right, threads - (threads / 2), tempArray);
            }
            mergeParallel(array, left, center, right, tempArray);
        }
    }
}


//############# Bitonic Sort Utils  #############
void bitonicSortParallel(int32_t* array, const size_t& start, const size_t& size, bool dir, const int& threads) {
    if (threads == 1) {
        bitonicSort(array, start, size, dir);
    }
    else {
        if (size > 1) {
            size_t center = lessPower2Than(size);
            #pragma omp parallel sections num_threads(2)
            {
            #pragma omp section
                bitonicSortParallel(array, start, center, !dir, threads / 2);
            #pragma omp section
                bitonicSortParallel(array, start + center, size - center, dir, threads - threads / 2);
            }
            bitonicMerge(array, start, size, dir);
        }
    }
}

void bitonicSort(int32_t* array, const size_t& start, const size_t& size, bool dir) {
    if (size > 1) {
        size_t center = lessPower2Than(size);
        bitonicSort(array, start, center, !dir);
        bitonicSort(array, start + center, size - center, dir);
        bitonicMerge(array, start, size, dir);
    }
}

void bitonicMerge(int32_t* array, const size_t& start, const size_t& size, bool dir) {
    if (size > 1) {
        size_t center = lessPower2Than(size);
        for (size_t i = start; i < start + size - center; i++)
            if (dir == (array[i] > array[i + center]))
                std::swap(array[i], array[i + center]);
        bitonicMerge(array, start, center, dir);
        bitonicMerge(array, start + center, size - center, dir);
    }
}



size_t lessPower2Than(const size_t& size) {
    size_t k = 1;
    while (k > 0 && k < size)
        k = k << 1;
    return k >> 1;
}


//#############  isSorted #############
bool isSorted(int32_t* array, const size_t& size) {
    for (size_t i = 0; i < size - 1; i++) {
        if (array[i] > array[i + 1]) {
            return false;
        }
    }
    return true;
}

//#############  chrono #############
template<typename F>
double benchmark(F& lambda) {
    auto start = std::chrono::high_resolution_clock::now();
    lambda();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    return elapsed.count();
}

//#############  generate array random ints #############
int32_t* generateArray(const size_t& size) {
    int32_t* array = new int32_t[size];
    srand(2020); 
    for (size_t i = 0; i < size; i++) {
        array[i] = rand();
    }
    return array;
}

//#############  read array from file #############
int32_t* readArray(const string& path, const size_t& size) {
    std::fstream file(path, std::ios::in);
    int32_t* array = new int32_t[size];
    for (size_t i = 0; i < size; i++) {
        file >> (int)array[i];
    }
    file.close();
    return array;
}

//#############  write array to file #############
void writeArray(const string& path, const size_t& size, int32_t* array) {
    std::fstream file(path, std::ios::out);
    for (size_t i = 0; i < size; i++) {
        file << (int)array[i] << endl;
    }
    file.close();
}