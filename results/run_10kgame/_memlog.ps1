param([string]$logPath = "C:\Users\hunte\projects\baseball\results\run_10kgame\_memlog.txt")
while ($true) {
  $p = Get-Process -Name python -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
  if ($p) {
    $rssMB = [math]::Round($p.WorkingSet64 / 1MB, 1)
    $ts = Get-Date -Format "HH:mm:ss"
    "$ts  pid=$($p.Id)  rss_mb=$rssMB" | Out-File -FilePath $logPath -Append -Encoding utf8
  }
  Start-Sleep -Seconds 30
}
