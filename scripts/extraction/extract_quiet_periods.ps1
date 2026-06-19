# scripts/extract_quiet_periods.ps1

$ErrorActionPreference = "Stop"

# À exécuter depuis la racine du projet :
# D:\Formation_Data_Engineer\Projets_certification\projet_final\Vulcadata

$Network = "PF"
$Stations = "CSS,DSO,ENO,FJS,HIM,SNE"
$Channels = "HHZ,EHZ,HHE,HHN"
$OutputDir = "data\extraction"
$S3Bucket = "vulcadata"

$QuietPeriods = @(
    @{
        Id = "quiet_2016_02_10"
        Start = "2016-02-10T20:11:30Z"
        End = "2016-02-13T20:11:30Z"
    },
    @{
        Id = "quiet_2016_11_23"
        Start = "2016-11-23T07:59:00Z"
        End = "2016-11-26T07:59:00Z"
    },
    @{
        Id = "quiet_2017_12_13"
        Start = "2017-12-13T14:50:00Z"
        End = "2017-12-16T14:50:00Z"
    },
    @{
        Id = "quiet_2018_12_24"
        Start = "2018-12-24T02:54:00Z"
        End = "2018-12-27T02:54:00Z"
    }
)

foreach ($Period in $QuietPeriods) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Extraction période calme : $($Period.Id)"
    Write-Host "Début : $($Period.Start)"
    Write-Host "Fin   : $($Period.End)"
    Write-Host "============================================================"
    Write-Host ""

    python scripts/extract_filter_aggregate_pf.py `
        --eruption-id $Period.Id `
        --network $Network `
        --stations $Stations `
        --channels $Channels `
        --starttime $Period.Start `
        --endtime $Period.End `
        --output-dir $OutputDir `
        --timeout 1200 `
        --upload-s3 `
        --s3-bucket $S3Bucket

    Write-Host ""
    Write-Host "Terminé : $($Period.Id)"
    Write-Host ""
}

Write-Host ""
Write-Host "Toutes les périodes calmes ont été extraites."