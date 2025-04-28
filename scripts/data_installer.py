# -*- coding: utf-8 -*-
# @Author: H. T. Duong Vu

import argparse
import requests
import os
import sys
import time

def download_file(url, output_path):
    # Get the start time for calculating download speed
    start_time = time.time()
    
    # Create a request with stream enabled
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        
        # Get the total file size if available
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        # Open the output file
        with open(output_path, "wb") as file:
            # Initialize progress variables
            last_percent = 0
            last_update_time = start_time
            
            # Download the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    # Calculate percentage and update progress
                    if total_size > 0:
                        percent = int(downloaded * 100 / total_size)
                        
                        # Update progress every 1% or at least every 3 seconds
                        current_time = time.time()
                        if percent > last_percent or current_time - last_update_time >= 3:
                            elapsed_time = current_time - start_time
                            speed = downloaded / (elapsed_time if elapsed_time > 0 else 1)
                            
                            # Format sizes for display
                            downloaded_str = format_size(downloaded)
                            total_str = format_size(total_size) if total_size > 0 else "Unknown"
                            speed_str = format_size(speed) + "/s"
                            
                            # Print progress
                            progress_bar = get_progress_bar(percent)
                            sys.stdout.write(f"\r{progress_bar} {percent}% | {downloaded_str}/{total_str} | {speed_str}")
                            sys.stdout.flush()
                            
                            last_percent = percent
                            last_update_time = current_time
                    else:
                        # If total size is unknown, just show downloaded amount
                        current_time = time.time()
                        if current_time - last_update_time >= 3:
                            elapsed_time = current_time - start_time
                            speed = downloaded / (elapsed_time if elapsed_time > 0 else 1)
                            
                            downloaded_str = format_size(downloaded)
                            speed_str = format_size(speed) + "/s"
                            
                            sys.stdout.write(f"\rDownloaded: {downloaded_str} | Speed: {speed_str}")
                            sys.stdout.flush()
                            
                            last_update_time = current_time
    
    # Calculate final stats
    elapsed_time = time.time() - start_time
    speed = downloaded / (elapsed_time if elapsed_time > 0 else 1)
    downloaded_str = format_size(downloaded)
    speed_str = format_size(speed) + "/s"
    
    # Print final message with newline
    print(f"\nDownload complete: {output_path}")
    print(f"Total downloaded: {downloaded_str} | Average speed: {speed_str} | Time: {format_time(elapsed_time)}")

def format_size(size_bytes):
    """Format bytes into a human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def format_time(seconds):
    """Format seconds into a human-readable time format"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        sec = seconds % 60
        return f"{int(minutes)} min {int(sec)} sec"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        sec = seconds % 60
        return f"{int(hours)} hr {int(minutes)} min {int(sec)} sec"

def get_progress_bar(percent, length=30):
    """Create a text-based progress bar"""
    filled_length = int(length * percent // 100)
    bar = '█' * filled_length + '░' * (length - filled_length)
    return f"[{bar}]"

def main():
    parser = argparse.ArgumentParser(description="Download a file from a URL with progress indication.")
    parser.add_argument("--url", required=True, help="The URL of the file to download.")
    parser.add_argument("--output", required=True, help="The path to save the downloaded file.")
    
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Print download information
    print(f"Downloading: {args.url}")
    print(f"Output path: {args.output}")
    
    try:
        download_file(args.url, args.output)
    except Exception as e:
        print(f"\nError during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()