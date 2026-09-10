# CWRU bearing dataset downloader (mirror: s-whynot/CWRU-dataset on GitHub)
# Downloads SKF 6205 drive-end (DE) data: normal (48k) + IR/OR@6/B at 3 severities (12k)
# Usage:  powershell -NoProfile -ExecutionPolicy Bypass -File tools\download_cwru.ps1
# Output: data\cwru\*.mat  (ASCII-only messages for PS 5.1 compatibility)

$ErrorActionPreference = "Stop"
$headers = @{ "User-Agent" = "dsh" }
$base = "12k_Drive_End_Bearing_Fault_Data"

$files = @(
    "Normal/97_Normal_0.mat", "Normal/98_Normal_1.mat", "Normal/99_Normal_2.mat",
    "IR/007/105_0.mat", "IR/007/106_1.mat", "IR/007/107_2.mat", "IR/007/108_3.mat",
    "IR/014/169_0.mat", "IR/014/170_1.mat", "IR/014/171_2.mat", "IR/014/172_3.mat",
    "IR/021/209_0.mat", "IR/021/210_1.mat", "IR/021/211_2.mat", "IR/021/212_3.mat",
    "B/007/118_0.mat", "B/007/119_1.mat", "B/007/120_2.mat", "B/007/121_3.mat",
    "B/014/185_0.mat", "B/014/186_1.mat", "B/014/187_2.mat", "B/014/188_3.mat",
    "B/021/222_0.mat", "B/021/223_1.mat", "B/021/224_2.mat", "B/021/225_3.mat",
    "OR/007/@6/130@6_0.mat", "OR/007/@6/131@6_1.mat", "OR/007/@6/132@6_2.mat", "OR/007/@6/133@6_3.mat",
    "OR/014/197@6_0.mat", "OR/014/198@6_1.mat", "OR/014/199@6_2.mat", "OR/014/200@6_3.mat",
    "OR/021/@6/234_0.mat", "OR/021/@6/235_1.mat", "OR/021/@6/236_2.mat", "OR/021/@6/237_3.mat"
)

$outDir = "data\cwru"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# raw 直链下载（不受 GitHub API 每小时 60 次匿名限流影响）
$rawBase = "https://raw.githubusercontent.com/s-whynot/CWRU-dataset/main"

$ok = 0
$failed = @()
foreach ($f in $files) {
    # Normal 目录位于仓库根目录；故障数据位于 12k_Drive_End_* 下
    if ($f -like "Normal/*") { $rel = $f } else { $rel = "$base/$f" }
    $url = "$rawBase/$rel"
    try {
        $leaf = Split-Path $f -Leaf
        $local = Join-Path $outDir $leaf
        Invoke-WebRequest -Uri $url -OutFile $local -UseBasicParsing -Headers $headers
        $size = (Get-Item $local).Length
        if ($size -lt 500000) { throw "file too small ($size B)" }
        $mb = [math]::Round($size / 1MB, 1)
        Write-Host ("[OK]   {0}  ({1} MB)" -f $leaf, $mb)
        $ok++
    } catch {
        $failed += $f
        Write-Host ("[FAIL] {0} : {1}" -f $f, $_.Exception.Message)
    }
}
Write-Host ("================ done: {0} ok, {1} failed ================" -f $ok, $failed.Count)
if ($failed.Count -gt 0) { Write-Host ("failed list: " + ($failed -join ", ")) }
