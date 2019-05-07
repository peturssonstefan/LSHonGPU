param(
    $dataset = "sample",
    $implementation = "2",
    $k = "32",
    $validate = "1",
    $writeResults = "0",
    $distanceFunc = "2",
    $sketchDim = "4",
    $compile = $True,
    $queueSize = '8',
    $framework = '0',
    $bucketKeyBits = '16',
    $tables = 1,
    $keysImplementation = 3,
    $withTQorBuffer = 0,
    $runWithSketchedData = 0,
    $resultFile = "..\resultsTest.txt"
)

.\changeVariables -queueSize $queueSize -withTQorBuffer $withTQorBuffer

if($compile){
    nvcc -rdc=true -O3 -arch=sm_61 -o knn kernel.cu gloveparser.cu resultWriter.cpp validation.cpp statisticsCpu.cpp randomVectorGenerator.cpp cudaHelpers.cu simHash.cu simpleLinearScan.cu optimizedLinearScan.cu memOptimizedLinearScan.cu
}


$fileexe = ".\knn.exe"

Write-Host "Running program" -ForegroundColor Green
Write-Host "$($fileexe) ..\datasets\$($dataset)_data.txt ..\datasets\$($dataset)_queries.txt ..\datasets\$($dataset)_$($distanceFunc)_validation1024k.txt $($validate) $($writeResults) $($k) $($implementation) $($sketchDim) $($distanceFunc) $($framework) $($bucketKeyBits) $($tables) $($keysImplementation) $($runWithSketchedData) $($resultFile)" -ForegroundColor Green

& $fileexe "..\datasets\$($dataset)_data.txt" "..\datasets\$($dataset)_queries.txt" "..\datasets\$($dataset)_$($distanceFunc)_validation1024k.txt" $validate $writeResults $k $implementation $sketchDim $distanceFunc $framework $bucketKeyBits $tables $keysImplementation $runWithSketchedData $resultFile