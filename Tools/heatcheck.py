import subprocess
import time
import csv
from datetime import datetime
import statistics
import os

# Function to run a shell command and capture its output
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr)
        raise Exception(f"Command failed: {command}")
    return result.stdout.strip()

# Function to get temperature metrics using rocm-smi
def get_temperature_metrics():
    try:
        output = run_command("rocm-smi --showtemp --csv")
        
        # Extract temperature from the CSV output
        lines = output.split('\n')
        
        metrics = {}
        
        for line in lines:
            if 'Temperature (Sensor)' in line:
                continue  # Skip header
            fields = line.split(',')
            if len(fields) > 1 and fields[1].strip() != 'N/A':
                try:
                    device_id = fields[0].strip()
                    temperature = float(fields[1].strip())
                    metrics[device_id] = {'temperature': f"{temperature} °C"}
                except ValueError:
                    pass  # Skip line due to conversion error
        
        return metrics
    except Exception as e:
        return {}

# Function to get power metrics using rocm-smi
def get_power_metrics():
    try:
        output = run_command("rocm-smi --showpower --csv")
        
        # Extract power from the CSV output
        lines = output.split('\n')
        
        metrics = {}
        
        for line in lines:
            if 'Graphics Package Power' in line or line.strip() == '':
                continue  # Skip header and empty lines
            fields = line.split(',')
            if len(fields) > 1:
                try:
                    device_id = fields[0].strip()
                    avg_power = fields[1].strip() if len(fields) > 1 else 'N/A'
                    current_power = fields[2].strip() if len(fields) > 2 else 'N/A'
                    
                    power = None
                    if avg_power != 'N/A':
                        power = float(avg_power)
                    elif current_power != 'N/A':
                        power = float(current_power)
                    
                    if power is not None:
                        metrics[device_id] = {'power': f"{power} W"}
                except ValueError:
                    pass  # Skip line due to conversion error
        
        return metrics
    except Exception as e:
        return {}

# Function to combine temperature and power metrics
def combine_metrics(temp_metrics, power_metrics):
    combined_metrics = {}
    for device_id, temp in temp_metrics.items():
        combined_metrics[device_id] = {'temperature': temp['temperature'], 'power': power_metrics.get(device_id, {}).get('power')}
    for device_id, power in power_metrics.items():
        if device_id not in combined_metrics:
            combined_metrics[device_id] = {'temperature': temp_metrics.get(device_id, {}).get('temperature'), 'power': power['power']}
    return combined_metrics

# Function to calculate average and maximum metrics
def calculate_statistics(metrics):
    stats = {}
    for device_id in set(metric['device_id'] for metric in metrics):
        device_metrics = [metric for metric in metrics if metric['device_id'] == device_id]
        temperatures = [float(metric['temperature'].split()[0]) for metric in device_metrics if metric['temperature'] is not None]
        powers = [float(metric['power'].split()[0]) for metric in device_metrics if metric['power'] is not None]
        
        stats[device_id] = {
            'avg_temperature (°C)': f"{round(statistics.mean(temperatures), 3)} °C" if temperatures else None,
            'max_temperature (°C)': f"{round(max(temperatures), 3)} °C" if temperatures else None,
            'avg_power (W)': f"{round(statistics.mean(powers), 3)} W" if powers else None,
            'max_power (W)': f"{round(max(powers), 3)} W" if powers else None
        }
    return stats

# Function to save statistics to a CSV file
def save_statistics_to_csv(stats, filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['device_id', 'avg_temperature (°C)', 'max_temperature (°C)', 'avg_power (W)', 'max_power (W)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for device_id, stat in stats.items():
            row = {'device_id': device_id}
            for key, value in stat.items():
                row[key] = value
            writer.writerow(row)

def main():
    global stop_collection
    stop_collection = False

    # Print message to recommend starting the benchmark
    print("For accurate response, it is recommended to begin the benchmark before initiating metrics collection.")
    
    # Ask the user for the duration of the recording session
    duration = float(input("Enter the duration of the recording session in seconds: "))
    
    # Ask the user for the benchmark name
    benchmark_name = input("Enter the benchmark name: ")
    
    # Confirm to begin metrics collection
    start_collection = input("Do you want to begin metrics collection? (y/n): ").strip().lower()
    
    if start_collection != 'y':
        print("Metrics collection aborted.")
        return
    
    # Generate a timestamped filename with the benchmark name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_filename = f'{benchmark_name}_gpu_stats_{timestamp}.csv'
    
    # Collect metrics
    all_metrics = []
    start_time = time.time()
    interval = 0.5  # Fixed interval of 0.5 seconds
    
    while time.time() - start_time < duration:
        temp_metrics = get_temperature_metrics()
        power_metrics = get_power_metrics()
        metrics = combine_metrics(temp_metrics, power_metrics)
        timestamp = time.time()
        for device_id, metric in metrics.items():
            all_metrics.append({'timestamp': timestamp, 'device_id': device_id, 'temperature': metric['temperature'], 'power': metric['power']})
        
        # Update display using ANSI escape codes
        os.system('clear')
        print("For accurate response, it is recommended to begin the benchmark before initiating metrics collection.")
        print(f"Enter the duration of the recording session in seconds: {duration}")
        print(f"Enter the benchmark name: {benchmark_name}")
        line = 4
        for device_id, metric in metrics.items():
            print(f"Recorded metrics at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}: Device ID={device_id}, Temperature={metric['temperature']}, Power={metric['power']}")
            line += 1
        
        time.sleep(interval)
    
    # Calculate and save statistics
    stats = calculate_statistics(all_metrics)
    save_statistics_to_csv(stats, stats_filename)
    
    # Print the statistics
    os.system('clear')
    print(f"Statistics saved to {stats_filename}")
    for device_id, stat in stats.items():
        print(f"Device ID={device_id} - Avg Temperature: {stat['avg_temperature (°C)']}, Max Temperature: {stat['max_temperature (°C)']}, Avg Power: {stat['avg_power (W)']}, Max Power: {stat['max_power (W)']}")

if __name__ == "__main__":
    main()
