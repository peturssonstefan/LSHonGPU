#!/bin/bash

echo "Running experiments"

dataset=$1
resultFile=$2

run_program(){
    implementation=$1
    k=$2
    validate="1"
    writeResults="0"
    distanceFunc=$3
    sketchDim=$4
    framework=$5
    bucketKeyBits=$6
    tables=$7
    keysImplementation=$8
    withSketchedData=$9

    echo "Run with ../datasets/${dataset}_data.txt ../datasets/${dataset}_queries.txt ../datasets/${dataset}_${distanceFunc}_validation1024k.txt $validate $writeResults $k $implementation $sketchDim $distanceFunc $framework $bucketKeyBits $tables $keysImplementation $runWithSketchedData $resultFile"
    ./knn "../datasets/${dataset}_data.txt" "../datasets/${dataset}_queries.txt" "../datasets/${dataset}_${distanceFunc}_validation${k}k.txt" $validate $writeResults $k $implementation $sketchDim $distanceFunc $framework $bucketKeyBits $tables $keysImplementation $withSketchedData $resultFile
    return
}

compile_program(){
    echo "Compiling program"
    nvcc -rdc=true -O3 -arch=sm_61 -o knn kernel.cu gloveparser.cu resultWriter.cpp validation.cpp statisticsCpu.cpp randomVectorGenerator.cpp cudaHelpers.cu simHash.cu simpleLinearScan.cu optimizedLinearScan.cu memOptimizedLinearScan.cu
}

change_constants(){
    queueSize=$1
    TQorBuf=$2

    sed -i -- "s/#define THREAD_QUEUE_SIZE [0-9][0-9]*/#define THREAD_QUEUE_SIZE $queueSize/g" constants.cuh

    sed -i -- "s/#define WITH_TQ_OR_BUFFER [0-9][0-9]*/#define WITH_TQ_OR_BUFFER $TQorBuf/g" constants.cuh
}

run_memOptimized(){
    queueSize=$1
    maxK=$(($queueSize*32))
    for ((k=32; k<=1024; k*=2))
    do
        if [ $k -gt $maxK ]
        then
            break
        fi
        run_program 2 $k 1 0 0 0 0 0 0
    done
    return
}

run_sketches(){
    queueSize=$1
    maxK=$(($queueSize*32))

    distanceFunc=1
    # Run simhash and one bit min hash
    for implementation in 3 5
    do
        for sketchDim in {4..16..2}
        do
            for ((k=32; k<=1024; k*=2))
            do
                if [ $k -gt $maxK ]
                then
                    break
                fi
                run_program $implementation $k $distanceFunc $sketchDim 0 0 0 0 0
            done
        done
        distanceFunc=2
    done

    distanceFunc=2
    # Run minhash
    for sketchDim in {32..64..8}
    do
        for ((k=32; k<=1024; k*=2))
        do
            if [ $k -gt $maxK ]
            then
                break
            fi
            run_program 4 $k $distanceFunc $sketchDim 0 0 0 0 0
        done
    done

    # Run Jonson lindenstrauss
    distanceFunc=3
    for sketchDim in {10..25..5}
    do
        for ((k=32; k<=1024; k*=2))
        do
            if [ $k -gt $maxK ]
            then
                break
            fi
            run_program 6 $k $distanceFunc $sketchDim 0 0 0 0 0
        done
    done
} 

run_lsh(){
    queueSize=$1
    maxK=$(($queueSize*32))

    for numTables in {2..10..2}
    do
        for bucketKeyBits in 4 8 16
        do
            for bucketKeyImplementation in 3 4 5
            do
                for sketchDim in {10..16..2}
                do
                    for ((k=32; k<=1024; k*=2))
                    do
                        if [ $k -gt $maxK ]
                        then
                            break
                        fi
                        run_program 3 $k 1 $sketchDim 1 $bucketKeyBits $numTables $bucketKeyImplementation 1
                        run_program 5 $k 2 $sketchDim 1 $bucketKeyBits $numTables $bucketKeyImplementation 1
                    done
                done

                for ((k=32; k<=1024; k*=2))
                do
                    if [ $k -gt $maxK ]
                    then
                        break
                    fi
                    run_program 2 $k 2 $sketchDim 1 $bucketKeyBits $numTables $bucketKeyImplementation 0
                done

            done
        done
    done
}

for ((queueSize=4; queueSize <= 128; queueSize*=2))
do
    #Change queueSize 
    #Change to buffer
    change_constants $queueSize 0
    #Compile
    compile_program
    #Run
    run_memOptimized $queueSize
    #run_sketches $queueSize
    #run_lsh $queueSize
done