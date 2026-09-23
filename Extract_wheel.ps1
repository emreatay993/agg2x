$wheel = '.\pyside6_essentials-6.11.1-cp310-abi3-win_amd64.whl'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $wheel).Path)

try {
    $entry = $archive.GetEntry('PySide6/Qt6QuickControls2FluentWinUI3StyleImpl.dll')
    if ($null -eq $entry) { throw 'DLL not found in wheel' }
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile(
        $entry,
        (Join-Path (Get-Location) 'Qt6QuickControls2FluentWinUI3StyleImpl.dll')
    )
} finally {
    $archive.Dispose()
}
