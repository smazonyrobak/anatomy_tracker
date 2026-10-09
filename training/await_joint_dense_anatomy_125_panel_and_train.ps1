$ErrorActionPreference = 'Stop'
$checkout = 'I:\AnatomyTracker\agent_worktrees\joint_v6_integration'
$panel = 'I:\AnatomyTracker\data\joint_dense_anatomy_125_dev_panel'
$trainRun = 'I:\AnatomyTracker\runs\joint_dense_anatomy_125'
$process = Get-Process -Id 12656 -ErrorAction SilentlyContinue
if ($process) {
    if ([Math]::Abs(($process.StartTime - [datetime]'2026-10-09T22:52:40').TotalSeconds) -gt 2) {
        throw 'Panel PID was reused.'
    }
    $process.WaitForExit()
    if ($null -ne $process.ExitCode -and $process.ExitCode -ne 0) {
        throw "Panel process exited $($process.ExitCode)."
    }
}

$done = Get-Content -LiteralPath (Join-Path $panel 'completed.json') -Raw | ConvertFrom-Json
$protocol = Get-Content -LiteralPath (Join-Path $panel 'protocol.json') -Raw | ConvertFrom-Json
$records = @(Get-Content -LiteralPath (Join-Path $panel 'records.jsonl') | ConvertFrom-Json)
if ($done.physical_sections -ne 256 -or $records.Count -ne 256 -or $done.synthetic_subjects -ne 8) {
    throw 'Incomplete panel.'
}
if ((Get-FileHash -LiteralPath (Join-Path $panel 'protocol.json') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $done.protocol_sha256 -or
    (Get-FileHash -LiteralPath (Join-Path $panel 'records.jsonl') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $done.records_sha256) {
    throw 'Panel receipt hash mismatch.'
}
if ((Get-FileHash -LiteralPath (Join-Path $checkout 'docs\publication\JOINT_DENSE_ANATOMY_125_PROTOCOL_20261009.md') -Algorithm SHA256).Hash.ToLowerInvariant() -ne $protocol.dense_125_protocol_sha256) {
    throw 'Training protocol changed while the panel was generated.'
}
foreach ($entry in $protocol.source_sha256.PSObject.Properties) {
    $path = Join-Path (Join-Path $panel 'source') $entry.Name
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.Value) {
        throw "Panel source hash mismatch: $($entry.Name)"
    }
}
foreach ($row in $records) {
    if ((Get-FileHash -LiteralPath (Join-Path $panel $row.file) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) {
        throw "Panel section hash mismatch: $($row.file)"
    }
}
if (@($records.panel_physical_section_id | Select-Object -Unique).Count -ne 256 -or
    @($records.section_id | Select-Object -Unique).Count -ne 256 -or
    @($records | Group-Object synthetic_subject_plan_id).Count -ne 8 -or
    @($records | Group-Object synthetic_subject_plan_id | Where-Object Count -ne 32).Count -ne 0 -or
    @($records | Where-Object eligible).Count -ne $done.eligible) {
    throw 'Panel identity, plan-count, or eligibility audit failed.'
}
if (Test-Path -LiteralPath $trainRun) { throw 'Training output already exists.' }
$python = 'I:\AtlasJointProject\envs\npixel_analysis\python.exe'
$training = Start-Process -FilePath $python -ArgumentList '-u', '-m', 'training.train_joint_dense_anatomy_125' `
    -WorkingDirectory $checkout -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput 'I:\AnatomyTracker\runs\joint_dense_anatomy_125_stdout.log' `
    -RedirectStandardError 'I:\AnatomyTracker\runs\joint_dense_anatomy_125_stderr.log'
"panel_verified=true training_pid=$($training.Id)"
