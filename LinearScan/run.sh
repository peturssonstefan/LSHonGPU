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

    echo "Run with ../datasets/${dataset}_data.txt ../datasets/${dataset}_queries.txt ../datasets/${dataset}_${distanceFunc}_validation1024k.txt $validate $writeResults $k $implementation $sketchDim $distanceFunc $framework $bucketKeyBits $tables $keysImplementation $withSketchedData $resultFile"
    ./knn "../datasets/${dataset}_data.txt" "../datasets/${dataset}_queries.txt" "../datasets/${dataset}_${distanceFunc}_validation1024k.txt" $validate $writeResults $k $implementation $sketchDim $distanceFunc $framework $bucketKeyBits $tables $keysImplementation $withSketchedData $resultFile
    return
}

compile_program(){
    echo "Compiling program"
    nvcc -std=c++11 -rdc=true -O3 -arch=sm_61 -o knn kernel.cu gloveparser.cu resultWriter.cpp validation.cpp statisticsCpu.cpp randomVectorGenerator.cpp cudaHelpers.cu simHash.cu simpleLinearScan.cu optimizedLinearScan.cu memOptimizedLinearScan.cu &> compile_log.txt
    errors=$(grep -c "error" "compile_log.txt")
    if [[ $errors -gt 0 ]]
    then 
	echo "Error in compilation"
	exit
    fi
    echo "Done compiling"
}

change_constants(){
    queueSize=$1
    TQorBuf=$2

    sed -i -- "s/#define THREAD_QUEUE_SIZE [0-9][0-9]*/#define THREAD_QUEUE_SIZE $queueSize/g" constants.cuh

    sed -i -- "s/#define WITH_TQ_OR_BUFFER [0-9][0-9]*/#define WITH_TQ_OR_BUFFER $TQorBuf/g" constants.cuh
}

run_memOptimized(){
    queueSize=$1
    maxK=$((($queueSize*32)/2))
    for k in 1024
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
    maxK=$((($queueSize*32)/2))

    distanceFunc=1 #change back when running with simhash again
    # Run simhash and one bit min hash
    for implementation in 3 
    do
        for sketchDim in {4..32..4}
        do
            for k in 32
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
    #for sketchDim in {64..128..8}
    #do
    #    for k in 1024
    #    do
    #        if [ $k -gt $maxK ]
    #        then
    #            break
    #        fi
    #        run_program 4 $k $distanceFunc $sketchDim 0 0 0 0 0
    #    done
    #done

    # Run Jonson lindenstrauss
    distanceFunc=3
    #for sketchDim in {10..25..5}
    #do
    #    for ((k=32; k<=1024; k*=2))
    #    do
    #        if [ $k -gt $maxK ]
    #        then
    #            break
    #        fi
    #        run_program 6 $k $distanceFunc $sketchDim 0 0 0 0 0
    #    done
    #done
} 

run_lsh(){
    queueSize=$1
    maxK=$((($queueSize*32)/2))
    sketchDim=16    

    for numTables in {35..40..5}
    do
       # for bucketKeyBits in 16
       # do
       #     for bucketKeyImplementation in 3 4 5
       #     do
       #         for k in 32
       #         do
       #             if [ $k -gt $maxK ]
       #             then
       #                 break
       #             fi
       #		    if [ $bucketKeyImplementation -eq 3 ]
       #	    then 
       #		run_program 2 $k 1 $sketchDim 1 16 $numTables $bucketKeyImplementation 0
       #		run_program 3 $k 1 $sketchDim 1 16 $numTables $bucketKeyImplementation 1
       #
       #	    elif [ $bucketKeyImplementation -eq 4 ]
       #	    then 
			#run_program 2 $k 2 $sketchDim 1 8 $numTables $bucketKeyImplementation 0
			#run_program 5 $k 2 $sketchDim 1 8 $numTables $bucketKeyImplementation 1
		    #else		 
		#	run_program 2 $k 2 $sketchDim 1 16 $numTables $bucketKeyImplementation 0
		#        run_program 5 $k 1 $sketchDim 1 16 $numTables $bucketKeyImplementation 1
       		#    fi	
               # done
            #done
        #done
	for sketchDim in {32..32..4} 
	do 
	    for k in 32 
	    do
		if [ $k -gt $maxK ]
		then
		    break
		fi
		for bucketKeyBits in 10
		do
		     run_program 3 $k 1 $sketchDim 1 $bucketKeyBits $numTables 3 1
		     #run_program 3 $k 1 $sketchDim 1 11 $numTables 7 1
		     #run_program 5 $k 2 $sketchDim 1 11 $numTables 5 1
		     #run_program 5 $k 2 $sketchDim 1 8 $numTables 4 1
		done	
	    done
	done
	#run_program 2 32 2 0 1 8 $numTables 4 0
	#run_program 5 32 2 32 1 8 $numTables 4 1
    done
}

for queueSize in 128
do
    #Change queueSize 
    #Change to buffer
    change_constants $queueSize 0
    #Compile
    compile_program
    #Run
    echo "TQ: ${queueSize}"
    #run_memOptimized $queueSize
    #run_sketches $queueSize
    run_lsh $queueSize
done
