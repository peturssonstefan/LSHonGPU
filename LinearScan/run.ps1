param(
    $dataset = "sample",
    $implementation = "2",
    $k = "32",
    $validate = "1",
    $writeResults = "0"
)
nvcc -rdc=true -O3 -arch=sm_61 -o knn kernel.cu gloveparser.cu resultWriter.cpp validation.cpp simHash.cu simpleLinearScan.cu optimizedLinearScan.cu memOptimizedLinearScan.cu weightedMinHash.cu

$fileexe = ".\knn.exe"

Write-Host "Running program" -ForegroundColor Green

& $fileexe "..\datasets\$($dataset)_data.txt" "..\datasets\$($dataset)_queries.txt" "..\datasets\$($dataset)_validation$($k)k.txt" $validate $writeResults $k $implementation 4 