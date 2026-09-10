# CWRU extra data downloader: OR@3/@12 clock positions + 0.028" diameter faults
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File tools\download_cwru_extra.ps1
# Output: data\cwru\*.mat  (raw direct links, no GitHub API rate limit)

$ErrorActionPreference = "Stop"
$headers = @{ "User-Agent" = "dsh" }
$base = "12k_Drive_End_Bearing_Fault_Data"
$rawBase = "https://raw.githubusercontent.com/s-whynot/CWRU-dataset/main"

$files = @(
    # 外圈故障 · 时钟 3 点方向（007/021）
    "OR/007/@3/144_0.mat", "OR/007/@3/145_1.mat", "OR/007/@3/146_2.mat", "OR/007/@3/147_3.mat",
    "OR/021/@3/246_0.mat", "OR/021/@3/247_1.mat", "OR/021/@3/248_2.mat", "OR/021/@3/249_3.mat",
    # 外圈故障 · 时钟 12 点方向（007/021）
    "OR/007/@12/156_0.mat", "OR/007/@12/157_1.mat", "OR/007/@12/158_2.mat", "OR/007/@12/159_3.mat",
    "OR/021/@12/258_0.mat", "OR/021/@12/259_1.mat", "OR/021/@12/260_2.mat", "OR/021/@12/261_3.mat",
    # 0.028" 大缺陷（内圈/滚动体，官方编号 3001~3008）
    "IR/028/3001_0.mat", "IR/028/3002_1.mat", "IR/028/3003_2.mat", "IR/028/3004_3.mat",
    "B/028/3005_0.mat", "B/028/3006_1.mat", "B/028/3007_2.mat", "B/028/3008_3.mat"
)

$outDir = "data\cwru"
$ok = 0
$failed = @()
foreach ($f in $files) {
    $leaf = Split-Path $f -Leaf
    $local = Join-Path $outDir $leaf
    if (Test-Path $local) { Write-Host "[SKIP] $leaf (exists)"; continue }
    $url = "$rawBase/$base/$f"
    $downloaded = $false
    for ($attempt = 1; $attempt -le 4 -and -not $downloaded; $attempt++) {
        try {
            Invoke-WebRequest -Uri $url -OutFile $local -UseBasicParsing -Headers $headers -TimeoutSec 60
            $size = (Get-Item $local).Length
            if ($size -lt 500000) { throw "file too small ($size B)" }
            Write-Host ("[OK]   {0}  ({1:N1} MB)" -f $leaf, ($size / 1MB))
            $ok++
            $downloaded = $true
        } catch {
            if ($attempt -ge 4) {
                $failed += $f
                Remove-Item $local -ErrorAction SilentlyContinue
                Write-Host ("[FAIL] {0} : {1}" -f $f, $_.Exception.Message)
            } else {
                Write-Host ("[RETRY] {0} (attempt {1}: {2})" -f $leaf, $attempt, $_.Exception.Message)
                Start-Sleep -Seconds 3
            }
        }
    }
}
Write-Host ("================ done: {0} ok, {1} failed ================" -f $ok, $failed.Count)
if ($failed.Count -gt 0) { Write-Host ("failed list: " + ($failed -join ", ")) }
