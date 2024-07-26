import os
import requests
import base64
import time
import subprocess
import random
import argparse
import csv
from datetime import datetime

AUTOMATIC1111_REPO = 'https://github.com/AUTOMATIC1111/stable-diffusion-webui'
LOCALHOST_URL = "http://127.0.0.1:7860"
API_URL = f"{LOCALHOST_URL}/sdapi/v1/txt2img"
CHECK_INTERVAL = 5
OUTPUT_BASE_DIR = "outputs"
RETRY_LIMIT = 3  # Limit for retrying server launch

# List of 20 random prompts
PROMPTS = [
    "A futuristic cityscape",
    "A serene forest with a flowing river",
    "A bustling marketplace in an ancient city",
    "A mystical castle floating in the sky",
    "A vibrant underwater coral reef",
    "A spaceship landing on an alien planet",
    "A tranquil beach at sunset",
    "A snowy mountain range",
    "A lively carnival with colorful lights",
    "A peaceful meadow with wildflowers",
    "A dark and eerie haunted house",
    "A futuristic robot in a city",
    "A dragon flying over a village",
    "A majestic waterfall in a dense jungle",
    "A sci-fi laboratory with advanced technology",
    "A beautiful galaxy with swirling stars",
    "A knight in shining armor",
    "A futuristic skyline at night",
    "A quaint village in the countryside",
    "A magical forest with glowing mushrooms"
]

def check_installation():
    if os.path.exists('stable-diffusion-webui'):
        print("Automatic1111 found. Not cloning.")
        return True
    else:
        print("Automatic1111 not found. Now cloning.")
        return False

def install_automatic1111():
    print("Cloning Automatic1111 repository...")
    try:
        subprocess.run(['git', 'clone', AUTOMATIC1111_REPO], check=True)
        print("Cloning completed.")
        print("Running make clean...")
        os.chdir('stable-diffusion-webui')
        subprocess.run(['make', 'clean'], check=True)
        os.chdir('..')
    except subprocess.CalledProcessError as e:
        print(f"Error during cloning: {e}")

def check_for_updates():
    os.chdir('stable-diffusion-webui')
    try:
        subprocess.run(['git', 'fetch'], check=True)
        local_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        remote_commit = subprocess.check_output(['git', 'rev-parse', '@{u}']).strip()
        if local_commit != remote_commit:
            print("Updates available. Pulling latest changes...")
            subprocess.run(['git', 'pull'], check=True)
            print("Running make clean...")
            subprocess.run(['make', 'clean'], check=True)
        else:
            print("No updates available.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking for updates: {e}")
    finally:
        os.chdir('..')

def check_if_running():
    try:
        response = requests.get(LOCALHOST_URL)
        if response.status_code == 200:
            print("Automatic1111 is running, skipping start.")
            return True
    except requests.ConnectionError:
        return False

def wait_for_server():
    print("Waiting for the server to be ready...")
    while not check_if_running():
        time.sleep(CHECK_INTERVAL)
    print("Server is ready.")

def launch_automatic1111(retry_count=0):
    if retry_count >= RETRY_LIMIT:
        print("Reached maximum retry limit for launching the server.")
        return False

    os.chdir('stable-diffusion-webui')
    print("Starting Automatic1111 server")
    subprocess.Popen(['python3', 'launch.py', '--no-half', '--api'])

    try:
        wait_for_server()
        os.chdir('..')
        return True
    except Exception as e:
        print(f"Error launching server: {e}")
        os.chdir('..')
        time.sleep(5)  # Wait before retrying
        return launch_automatic1111(retry_count + 1)

def kill_server():
    print("Stopping the Automatic1111 server...")
    try:
        subprocess.run(['pkill', '-f', 'launch.py'], check=True)
        print("Server stopped.")
        time.sleep(5)  # Wait before starting the server again to avoid race conditions
    except subprocess.CalledProcessError as e:
        print(f"Error stopping the server: {e}")

def warm_up_server():
    print("Warming up the server...")
    prompt = "A warm-up prompt"
    payload = {
        "prompt": prompt,
        "steps": 50,
        "batch_size": 1
    }
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        print("Warm-up completed.")
    except requests.RequestException as e:
        print(f"Error during warm-up: {e}")

def save_image(image_data, output_dir, index):
    image_filename = os.path.join(output_dir, f"output_{index}.png")
    with open(image_filename, "wb") as file:
        file.write(image_data)
    return image_filename

def generate_images(prompt, num_images=1, steps=50, batch_number=1, seed=None):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, OUTPUT_BASE_DIR, f"output_{current_time}_batch_{batch_number}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    payload = {
        "prompt": prompt,
        "steps": steps,
        "batch_size": num_images
    }

    if seed is not None:
        payload["seed"] = seed

    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        end_time = time.time()
    except requests.RequestException as e:
        print(f"Error generating images: {e}")
        return

    inference_time = (end_time - start_time)
    iterations_per_second = (steps / inference_time) * num_images  # Multiply by batch size
    image_filenames = []

    if 'images' in result:
        for i, image in enumerate(result['images']):
            image_data = base64.b64decode(image)
            image_filenames.append(save_image(image_data, output_dir, i))
    else:
        print("No images found in the response")

    save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames, batch_number)

def save_metrics(output_dir, payload, inference_time, iterations_per_second, image_filenames, batch_number):
    metrics_filename = os.path.join(output_dir, "metrics.csv")
    file_exists = os.path.isfile(metrics_filename)
    metrics_header = [
        "Batch Number", "Prompt", "Steps", "Batch Size", "Seed", "Inference Time (seconds)", 
        "Iterations per Second", "Generated Images"
    ]
    metrics_data = [
        batch_number, payload['prompt'], payload['steps'], payload['batch_size'], payload.get('seed', 'N/A'),
        f"{inference_time:.2f}", f"{iterations_per_second:.2f}", 
        ", ".join(image_filenames)
    ]
    
    with open(metrics_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(metrics_header)
        writer.writerow(metrics_data)

    # Print metrics to console
    print(f"Metrics for batch {batch_number}:")
    for header, data in zip(metrics_header, metrics_data):
        print(f"{header}: {data}")
    print()

def run_in_loop(kill_flag, seed):
    while True:
        warm_up_server()  # Always run the warm-up batch

        # Generate images using different prompts in three batches
        used_prompts = set()
        batch_sizes = [1, 5, 10]
        for batch_number, num_images in enumerate(batch_sizes, 1):
            selected_prompt = random.choice([p for p in PROMPTS if p not in used_prompts])
            used_prompts.add(selected_prompt)
            print(f"Selected prompt for batch {batch_number}: {selected_prompt}")
            generate_images(selected_prompt, num_images=num_images, batch_number=batch_number, seed=seed)
        
        if kill_flag:
            kill_server()
            time.sleep(5)  # Wait before restarting the server to avoid race conditions
            launch_automatic1111()
        time.sleep(CHECK_INTERVAL)  # Wait before starting the next loop

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the script in a loop if the --loop flag is provided.")
    parser.add_argument('--loop', action='store_true', help="Run the script in a continuous loop.")
    parser.add_argument('--kill', action='store_true', help="Kill the server at the end of each run.")
    parser.add_argument('--seed', type=int, help="Seed value for image generation.")
    args = parser.parse_args()

    if not check_installation():
        install_automatic1111()
    else:
        check_for_updates()

    server_running = check_if_running()
    if not server_running:
        if not launch_automatic1111():
            print("Failed to launch the server.")
            exit(1)

    if args.loop:
        run_in_loop(args.kill, args.seed)
    else:
        warm_up_server()  # Always run the warm-up batch

        # Generate images using different prompts in three batches
        used_prompts = set()
        batch_sizes = [1, 5, 10]
        for batch_number, num_images in enumerate(batch_sizes, 1):
            selected_prompt = random.choice([p for p in PROMPTS if p not in used_prompts])
            used_prompts.add(selected_prompt)
            print(f"Selected prompt for batch {batch_number}: {selected_prompt}")
            generate_images(selected_prompt, num_images=num_images, batch_number=batch_number, seed=args.seed)
        
        if args.kill:
            kill_server()
