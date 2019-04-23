param(
    $queueSize = '4',
    $withTQorBuffer = '0'
)
$newFile = Get-Content -Path .\constants.cuh | ForEach-Object {$_ -replace '#define THREAD_QUEUE_SIZE \d+' , "#define THREAD_QUEUE_SIZE $($queueSize)" } | ForEach-Object {$_ -replace '#define WITH_TQ_OR_BUFFER \d+' , "#define WITH_TQ_OR_BUFFER $($withTQorBuffer)" }

$newFile

Set-Content -Path .\constants.cuh -Value $newFile