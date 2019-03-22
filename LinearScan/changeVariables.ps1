param(
    $queueSize = '4'
)
$newFile = Get-Content -Path .\constants.cuh | ForEach-Object {$_ -replace '#define THREAD_QUEUE_SIZE \d+' , "#define THREAD_QUEUE_SIZE $($queueSize)" }

$newFile

Set-Content -Path .\constants.cuh -Value $newFile