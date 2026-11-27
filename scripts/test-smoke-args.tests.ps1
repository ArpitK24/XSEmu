# Run with powershell.exe or pwsh.exe. Verifies smoke.ps1 preserves -Game paths with spaces and trailing backslashes.
$ErrorActionPreference = 'Stop'
$smoke = Join-Path $PSScriptRoot 'smoke.ps1'
$fixture = Join-Path ([IO.Path]::GetTempPath()) ('xcl-smoke-args-' + [guid]::NewGuid())
$null = New-Item -ItemType Directory -Path $fixture
$mockExe = Join-Path $fixture 'mock-xsemu.exe'
$logFile = Join-Path $fixture 'xcl.log'
$capturedFile = Join-Path $fixture 'captured-args.txt'

try {
    Add-Type -OutputAssembly $mockExe -OutputType ConsoleApplication -TypeDefinition @'
using System;
using System.IO;
using System.Text;

public static class Program {
    public static int Main(string[] args) {
        string dir = Path.GetDirectoryName(typeof(Program).Assembly.Location);
        File.WriteAllLines(Path.Combine(dir, "captured-args.txt"), args, Encoding.UTF8);
        string logPath = Path.Combine(dir, "xcl.log");
        File.WriteAllText(logPath, "(10 frames shown, 10 PresentX calls)\nactivity (2 s): 5 presents\nmemory: 512 MB committed\n", Encoding.UTF8);
        Console.WriteLine("Log:    " + logPath);
        return 0;
    }
}
'@

    $gamePath = Join-Path $fixture 'My Title\'
    $output = & $smoke -Game $gamePath -Seconds 5 -Xcl $mockExe
    if ($LASTEXITCODE -ne 0) { throw "smoke.ps1 exited $LASTEXITCODE" }

    $actualArgs = Get-Content -LiteralPath $capturedFile -Encoding UTF8
    if ($actualArgs.Count -ne 2 -or $actualArgs[0] -ne 'run' -or $actualArgs[1] -ne $gamePath) {
        throw "Expected args @('run', '$gamePath'), got: $($actualArgs -join ' | ')"
    }

    $text = ($output -join "`n")
    if (!$text.Contains('xcl-smoke v1') -or !$text.Contains('frames shown: 10 (PresentX 10)') -or !$text.Contains('peak commit: 512 MB')) {
        throw "Unexpected smoke.ps1 output:`n$text"
    }
    Write-Host 'PASS: smoke.ps1 preserved trailing backslash and produced xcl-smoke v1 block.'
} finally {
    if (Test-Path -LiteralPath $fixture) { Remove-Item -LiteralPath $fixture -Recurse -Force }
}
