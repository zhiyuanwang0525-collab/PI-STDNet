# Data Preparation

This repository does not include raw TorNet radar data because dataset access and redistribution are governed by the dataset license. The TorNet benchmark dataset is publicly available from the official TorNet data source. Users should follow the original dataset license and access instructions.

Prepare the data outside the source tree and point configs to it. Avoid placing the dataset inside this repository:

```text
/path/to/TorNet/
  catalog.csv
  tornet_2013/
    sample_file.nc
  tornet_2014/
    sample_file.nc
  ...
```

The catalog is expected to include at least:

- `type`: split label such as `train` or `test`
- `category`: `TOR`, `WRN`, `Tornado Warning`, or null/non-tornado category
- `filename`: relative NetCDF filename
- `start_time`: optional timestamp used to infer `tornet_YYYY/`

Update:

```yaml
data:
  dataset_root: "/path/to/TorNet"
  catalog_path: "/path/to/TorNet/catalog.csv"
```

Do not commit `.nc`, `.h5`, `.npy`, `.npz`, or generated data files.
