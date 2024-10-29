import logging
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)


def data_stream_simulation(num_points=1000, noise_level=0.35, anomaly_frequency=0.05, seasonality_period=50, seed=None):
    """Generates a data stream with random walk patterns, random noise, occasional anomalies, and seasonality."""
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points should be a positive integer.")
    if not (0 <= noise_level < 1):
        raise ValueError("noise_level should be between 0 and 1.")
    if not (0 <= anomaly_frequency <= 1):
        raise ValueError("anomaly_frequency should be between 0 and 1.")
    if not (seasonality_period > 0):
        raise ValueError("seasonality_period should be a positive integer.")

    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    else:
        seed = random.randint(1, 10000)  # Generate a random seed
        random.seed(seed)
        np.random.seed(seed)

    logging.info(f"Using seed: {seed}")

    data = []
    current_value = random.uniform(0, 10)  # Start with a random initial value

    for i in range(num_points):
        # Random walk step
        step = np.random.normal(0, noise_level)  # Add noise to the step
        current_value += step

        # Introduce seasonality
        seasonality = 5 * np.sin(2 * np.pi * i / seasonality_period)  # Sine wave for seasonality

        # Introduce anomalies randomly
        anomaly = random.choice([7, -7]) if random.random() < anomaly_frequency else 0

        # Create the data point with seasonality
        data_point = current_value + seasonality + anomaly
        data.append(data_point)

    return data


def detect_cycles(data, dynamic_min_frequency=0.1):
    """Detects cycles in the data using Fast Fourier Transform (FFT) with dynamic threshold and min frequency."""
    # Apply a Hanning window to the data
    windowed_data = np.hanning(len(data)) * data
    fft_result = np.fft.fft(windowed_data)
    frequencies = np.fft.fftfreq(len(data))
    magnitudes = np.abs(fft_result)

    # Log frequencies and magnitudes
    for f, mag in zip(frequencies, magnitudes):
        logging.debug(f"Frequency: {f}, Magnitude: {mag}")

    # Adjust the dynamic threshold based on the magnitude statistics
    mean_magnitude = np.mean(magnitudes)
    std_dev_magnitude = np.std(magnitudes)
    dynamic_threshold = mean_magnitude + 2 * std_dev_magnitude  # Example adjustment

    # Adjust dynamic minimum frequency based on recent frequency analysis
    recent_frequencies = frequencies[magnitudes > 0]
    if len(recent_frequencies) > 0:
        dynamic_min_frequency = np.min(
            np.abs(recent_frequencies)) * 1.1  # Slightly above the minimum detected frequency

    # Filter out low frequencies
    significant_frequencies = frequencies[
        (magnitudes > dynamic_threshold) & (np.abs(frequencies) > dynamic_min_frequency)]
    return significant_frequencies


def adaptive_z_score_anomaly_detector(window, primary_threshold=1.5, secondary_threshold=2.5):
    """Detects anomalies using adaptive Z-score thresholds with error handling."""
    if not window:
        raise ValueError("Window data cannot be empty.")
    if primary_threshold <= 0 or secondary_threshold <= 0:
        raise ValueError("Thresholds must be positive values.")

    mean = np.mean(window)
    std_dev = np.std(window)
    last_point = window[-1]

    if std_dev == 0:
        return False, 0  # Avoid division by zero
    z_score = (last_point - mean) / std_dev
    return (abs(z_score) > primary_threshold or abs(z_score) > secondary_threshold), z_score


def adaptive_ewma_anomaly_detector(data_point, alpha=0.3, base_threshold=2.0, scale_factor=1.5):
    """Detects anomalies using adaptive EWMA with dynamic threshold scaling."""
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if base_threshold <= 0 or scale_factor <= 0:
        raise ValueError("base_threshold and scale_factor must be positive values.")

    # Initialize the mean and std_dev if they don't exist
    if not hasattr(adaptive_ewma_anomaly_detector, 'mean'):
        adaptive_ewma_anomaly_detector.mean = data_point
        adaptive_ewma_anomaly_detector.std_dev = 0

    # Update EWMA mean and std_dev
    adaptive_ewma_anomaly_detector.mean = alpha * data_point + (1 - alpha) * adaptive_ewma_anomaly_detector.mean
    adaptive_ewma_anomaly_detector.std_dev = alpha * abs(data_point - adaptive_ewma_anomaly_detector.mean) + \
                                             (1 - alpha) * adaptive_ewma_anomaly_detector.std_dev

    # Calculate deviation and dynamic threshold
    deviation = abs(data_point - adaptive_ewma_anomaly_detector.mean)
    dynamic_threshold = base_threshold + scale_factor * adaptive_ewma_anomaly_detector.std_dev
    return deviation > dynamic_threshold, deviation


def log_anomalies_and_cycles(z_anomalies, ewma_anomalies, cycle_points):
    """Logs detected anomalies and cycles to the console."""
    if z_anomalies:
        logging.info(f"Z-score Anomalies Detected: {z_anomalies[-1]}")  # Log the last Z-score anomaly

    if ewma_anomalies:
        logging.info(f"EWMA Anomalies Detected: {ewma_anomalies[-1]}")  # Log the last EWMA anomaly

    if cycle_points:
        logging.info(f"Cycle Points Detected: {cycle_points[-1]}")  # Log the last cycle point


def visualize_data_stream(data_stream, window_size=50, z_primary=1.5, z_secondary=2.5, ewma_alpha=0.3,
                          ewma_base=1.5, ewma_scale=2.0, update_interval=20):
    plt.figure(figsize=(14, 8))
    plt.title("Data Stream Analysis with Anomaly Detection")  # Set a clear title
    plt.xlabel("Time (Data Points)")  # More descriptive x-axis label
    plt.ylabel("Value")  # y-axis label remains the same

    x_data, y_data = [], []
    z_anomalies, ewma_anomalies, cycle_points = [], [], []
    data_window = deque(maxlen=window_size)
    original_value = data_stream[0]  # Reference original value

    for i, point in enumerate(data_stream):
        x_data.append(i)
        y_data.append(point)
        data_window.append(point)

        # Detect cycles in the data every `update_interval` iterations
        if i > 1 and i % update_interval == 0:
            significant_frequencies = detect_cycles(y_data)
            for freq in significant_frequencies:
                if freq > 0:  # Only consider positive frequencies
                    period = int(1 / freq) if freq != 0 else 1
                    if period < len(data_window) and period not in cycle_points:  # Avoid duplicates
                        # Only add cycle points that are within the bounds of x_data
                        if i - period >= 0:
                            cycle_points.append(i - period)
                        if i + period < len(data_stream):
                            cycle_points.append(i + period)

        # Z-score anomaly detection
        is_z_outlier = False
        z_score = 0  # Initialize z_score for logging
        if len(data_window) == window_size:
            try:
                is_z_outlier, z_score = adaptive_z_score_anomaly_detector(data_window, primary_threshold=z_primary,
                                                                          secondary_threshold=z_secondary)
                if is_z_outlier:
                    z_anomalies.append((i, point))
            except ValueError as e:
                logging.error(f"Error in Z-score detection: {e}")

        # EWMA anomaly detection for non-outliers
        if not is_z_outlier:
            try:
                is_ewma_anomaly, _ = adaptive_ewma_anomaly_detector(point, alpha=ewma_alpha, base_threshold=ewma_base,
                                                                    scale_factor=ewma_scale)
                if is_ewma_anomaly:
                    ewma_anomalies.append((i, point))
            except ValueError as e:
                logging.error(f"Error in EWMA detection: {e}")

        # Update plot every `update_interval` iterations
        if i % update_interval == 0:
            log_anomalies_and_cycles(z_anomalies, ewma_anomalies, cycle_points)
            plt.clf()  # Clear the figure for fresh plotting

            # Set line color based on comparison to the original value
            line_color = 'green' if point > original_value else 'red'
            plt.plot(x_data, y_data, label="Data Stream", color=line_color, linewidth=1.5, alpha=0.8)

            # Highlighting Z-score anomalies in red
            if z_anomalies:
                x_z_anoms, y_z_anoms = zip(*z_anomalies)
                plt.scatter(x_z_anoms, y_z_anoms, color="red", s=80,
                            label="Outliers", edgecolor='black', marker='o')

            # Highlighting EWMA anomalies in black
            if ewma_anomalies:
                x_ewma_anoms, y_ewma_anoms = zip(*ewma_anomalies)
                plt.scatter(x_ewma_anoms, y_ewma_anoms, color="black", s=80,
                            label="Possible pattern deviations", edgecolor='black', marker='o')

            # Shade the area under the cycle in grey, in the background
            valid_cycle_points = [point for point in cycle_points if point < len(x_data)]
            if valid_cycle_points:
                for j in range(len(valid_cycle_points) - 1):
                    if j < len(valid_cycle_points) - 1:
                        plt.fill_between(x_data[valid_cycle_points[j]:valid_cycle_points[j + 1]],
                                         y_data[valid_cycle_points[j]:valid_cycle_points[j + 1]],
                                         color="grey", alpha=0.05)

            # Set the title and labels again
            plt.title("Data Stream Analysis with Anomaly Detection")  # Set a clear title
            plt.xlabel("Time (Data Points)")  # More descriptive x-axis label
            plt.ylabel("Value")  # y-axis label remains the same
            plt.grid(True)  # Add a grid for better readability
            plt.legend()

            # Add counts for Z-score and EWMA anomalies at the bottom of the plot
            z_count = len(z_anomalies)
            ewma_count = len(ewma_anomalies)
            plt.text(0.5, -0.1, f"Grey area implies that the data is following a pattern. Outliers: {z_count}, Pattern breakers: {ewma_count}",
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)

            plt.pause(0.1)  # Allow time for the plot to update
    logging.info("The data processing and image generation is complete. You can choose to close the chart when you please and the program will terminate")
    plt.show()
    plt.savefig('output_plot.png')

# Example usage with user input
if __name__ == "__main__":
    try:
        num_points = int(input("Enter the number of points (default 1000): ") or 1000)
        seed_input = input("Enter a seed for randomness (leave empty for random seed): ")
        seed = int(seed_input) if seed_input else None

        data_stream = data_stream_simulation(num_points=num_points, noise_level=0.35, anomaly_frequency=0.05,
                                             seasonality_period=50, seed=seed)
        visualize_data_stream(data_stream, window_size=50, z_primary=1.5, z_secondary=2.5,
                              ewma_alpha=0.3, ewma_base=1.5, ewma_scale=2.0, update_interval=5)
    except ValueError as e:
        logging.error(f"Input error: {e}")
    finally:
       logging.info("The program has terminated") 