# Benchmark Results Summary

## Complete Results

| model_size   |   d_model |   context_length |   num_layers |   num_heads |   batch_size | forward_only   |   mean_time_ms |   std_time_ms |
|:-------------|----------:|-----------------:|-------------:|------------:|-------------:|:---------------|---------------:|--------------:|
| small        |        64 |              128 |            6 |           2 |            4 | False          |          25.81 |          1.35 |
| xl           |       256 |              128 |            4 |           4 |            4 | False          |          22.18 |          0.67 |
| medium       |       256 |              128 |            4 |           4 |            4 | False          |          22.12 |          0.37 |
| large        |       256 |              128 |            4 |           4 |            4 | False          |          22.11 |          0.82 |
| small        |       256 |              128 |            4 |           4 |            4 | False          |          22.31 |          0.97 |
| 2.7B         |       256 |              128 |            4 |           4 |            4 | False          |          21.93 |          0.31 |
| xl           |       256 |              256 |            4 |           4 |            4 | False          |          46.01 |          0.17 |
| medium       |       256 |              256 |            4 |           4 |            4 | False          |          46.58 |          0.24 |
| large        |       256 |              256 |            4 |           4 |            4 | False          |          46.31 |          0.21 |
| small        |       256 |              256 |            4 |           4 |            4 | False          |          46.44 |          0.21 |
| 2.7B         |       256 |              256 |            4 |           4 |            4 | False          |          46.58 |          0.25 |

## Results by Model Size

### Model: small

|   context_length |   mean_time_ms |   std_time_ms |
|-----------------:|---------------:|--------------:|
|              128 |          25.81 |          1.35 |
|              128 |          22.31 |          0.97 |
|              256 |          46.44 |          0.21 |

### Model: xl

|   context_length |   mean_time_ms |   std_time_ms |
|-----------------:|---------------:|--------------:|
|              128 |          22.18 |          0.67 |
|              256 |          46.01 |          0.17 |

### Model: medium

|   context_length |   mean_time_ms |   std_time_ms |
|-----------------:|---------------:|--------------:|
|              128 |          22.12 |          0.37 |
|              256 |          46.58 |          0.24 |

### Model: large

|   context_length |   mean_time_ms |   std_time_ms |
|-----------------:|---------------:|--------------:|
|              128 |          22.11 |          0.82 |
|              256 |          46.31 |          0.21 |

### Model: 2.7B

|   context_length |   mean_time_ms |   std_time_ms |
|-----------------:|---------------:|--------------:|
|              128 |          21.93 |          0.31 |
|              256 |          46.58 |          0.25 |

## Summary Statistics

|       |   mean_time_ms |   std_time_ms |
|:------|---------------:|--------------:|
| count |          11    |         11    |
| mean  |          33.49 |          0.51 |
| std   |          12.39 |          0.39 |
| min   |          21.93 |          0.17 |
| 25%   |          22.15 |          0.23 |
| 50%   |          25.81 |          0.31 |
| 75%   |          46.38 |          0.75 |
| max   |          46.58 |          1.35 |
