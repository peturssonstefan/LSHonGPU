param(
    $dataset = "sample",
    $implementation = "2",
    $k = "32",
    $validate = "1",
    $writeResults = "0",
    $distanceFunc = "2",
    $bits = "4",
    $compile = $True,
    $queueSize = '4'
)

.\changeVariables -queueSize $queueSize

if($compile){
    nvcc -rdc=true -O3 -arch=sm_61 -o knn kernel.cu gloveparser.cu resultWriter.cpp validation.cpp cudaHelpers.cu simHash.cu simpleLinearScan.cu optimizedLinearScan.cu memOptimizedLinearScan.cu weightedMinHash.cu
}


$fileexe = ".\knn.exe"

Write-Host "Running program" -ForegroundColor Green
Write-Host "$($fileexe) ..\datasets\$($dataset)_data.txt ..\datasets\$($dataset)_queries.txt ..\datasets\$($dataset)_$($distanceFunc)_validation$($k)k.txt $($validate) $($writeResults) $($k) $($implementation) $($bits) $($distanceFunc)" -ForegroundColor Green

& $fileexe "..\datasets\$($dataset)_data.txt" "..\datasets\$($dataset)_queries.txt" "..\datasets\$($dataset)_$($distanceFunc)_validation$($k)k.txt" $validate $writeResults $k $implementation $bits $distanceFunc