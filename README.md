
# Efficient Data Stream Anomaly Detection for random data

This project is a Python-based tool for detecting anomalies in a data stream using Z-score and Exponentially Weighted Moving Average (EWMA) techniques. It simulates a dynamic data stream, detects anomalies, and visualizes cycles and anomaly points in real-time.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Motivation](#motivation)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Methods](#methods)
- [Example](#example)
- [Notes](#notes)

## Project Overview
This project is designed for real-time anomaly detection in data streams, such as IoT sensor data, financial time series, or network traffic. It generates a simulated data stream with patterns, noise, seasonality, and anomalies, which is then analyzed using Z-score and EWMA methods. The visualization is displayed using Matplotlib, allowing users to monitor the data stream and view detected anomalies in a GUI. The algorithm is also able to detect seasonalities and patterns in the random data if they appear.

### Features
- **Data Stream Simulation**: Generates a random walk data stream with noise, anomalies, and seasonality.
- **Z-score and EWMA Anomaly Detection**: Detects anomalies in real time using Z-score and adaptive EWMA techniques.
- **Cycle Detection**: Identifies cyclic patterns in the data using FFT.
- **Real-Time Visualization**: Displays a real-time plot of the data stream with dynamic graph coloring, anomaly and cycle markings.
- **Logging**: Provides logging for tracking key events and issues.

### Motivation
This anomaly and cycle detection algorithm was chosen for its adaptability and efficiency in handling real-time data streams, particularly beneficial in environments with fluctuating patterns. The combination of Z-score, Exponentially Weighted Moving Average (EWMA), and Fast Fourier Transform (FFT) techniques provides a robust, dual-layer approach to ensure sensitive and stable anomaly detection.

- Z-score Detection: Z-score quickly identifies outliers by highlighting data points that deviate significantly from the mean. This makes it suitable for real-time applications requiring immediate anomaly detection.

- EWMA Detection: EWMA adaptively smooths out noise while remaining responsive to trends, giving priority to recent data. This is valuable in non-stationary data streams where data patterns may shift over time, enabling the model to adjust dynamically.

- FFT-Based Cycle Detection: FFT isolates dominant frequencies within the data, dynamically filtering significant cycles. With adaptive thresholding and minimum frequency adjustments, it efficiently detects periodic patterns, reducing the influence of noise.

Together, these methods provide a comprehensive framework, balancing sensitivity and stability to ensure accurate, timely anomaly detection and cycle recognition across various real-time data patterns.

## Installation
Clone the repository:

```bash
git clone https://github.com/Yuvaraj0702/cobblestone_assessment.git
```
Navigate to working directory:

```bash
cd cobblestone_assessment
```
Install dependencies:

```bash
pip install -r requirements.txt
```

Note: This script requires `Matplotlib` and `NumPy`. Ensure you have them installed via requirements.txt or individually:

```bash
pip install matplotlib numpy
```

Tkinter: For interactive GUI visualization, ensure Tkinter is installed, as the Matplotlib TkAgg backend is used. Some devices may not have this installed especially linux machines. Below are the steps to install it.

```bash
sudo apt-get update
```
```bash
sudo apt-get install python3.11-tk
```
```bash
pip install matplotlib 
```

## Usage
Run the Main Script:

```bash
python anomaly_detector.py
```
Or

```bash
python3 anomaly_detector.py
```
Depending on your system.

The command line will ask for the number of data points you want to simulate first. You may enter any positive integer

Then the command line will ask for a random seed. This is for more randomness and repeatability of generated data, which may be useful at times.

The plot will then open in a separate window, updating in real-time. A log of all the anomalies and season detection is also present in the console.

### Script Arguments (optional): 
Modify parameters in the `data_stream_simulation` and `visualize_data_stream` functions to adjust data stream properties and detection thresholds.

## Parameters
This is for further testing beyond the normal use case where the tester can modify  parameters within the functions to view output.
### Data Simulation
- `num_points` (int): Number of data points to generate.
- `noise_level` (float): Amount of noise in the data (0–1).
- `anomaly_frequency` (float): Probability of generating an anomaly (0–1).

### Z-score Anomaly Detector
- `window` (deque): Sliding window of data points.
- `primary_threshold` (float): Z-score threshold for anomaly detection.
- `secondary_threshold` (float): Secondary threshold for higher sensitivity.

### EWMA Anomaly Detector
- `alpha` (float): Smoothing factor for EWMA (0–1).
- `base_threshold` (float): Baseline threshold for anomalies.
- `scale_factor` (float): Scaling factor for dynamic threshold adjustment.

## Methods
- **data_stream_simulation()**: Generates a synthetic data stream with noise, seasonality, and occasional anomalies based on specified parameters.
- **detect_cycles()**: Identifies cycles in the data stream by using the Fast Fourier Transform (FFT) with dynamic thresholding and filters low frequencies.
- **adaptive_z_score_anomaly_detector()**: Detects anomalies using the Z-score, adjusting thresholds for dynamic data windows.
- **adaptive_ewma_anomaly_detector()**: Uses the Exponentially Weighted Moving Average (EWMA) for adaptive anomaly detection.
- **visualize_data_stream()**: Plots the data stream and marks anomalies and cycles in real-time, pausing briefly after each point to simulate a continuous data stream.

## Example
```python
# Generate and visualize a data stream with the following parameters:
data_stream = data_stream_simulation(num_points=500, noise_level=0.3, anomaly_frequency=0.05)
visualize_data_stream(data_stream)
```

## Notes
- Multithreading could have sped up the processing (thread for console logging anomalies and another one for the visualization) but was not used due the requirement of limitted external libraries.
- Ensure Tkinter or an equivalent GUI backend is installed to support Matplotlib’s GUI rendering.
- The script logs anomalies and cycles to the console and in the Matplotlib window for easy monitoring.
- This project is structured to accommodate further improvements, such as integrating with a live data feed or customizing anomaly thresholds dynamically.

