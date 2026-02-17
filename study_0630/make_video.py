import cv2
import os
import glob
import argparse
from natsort import natsorted

def create_video_from_images(input_dir, output_path, fps=30, image_format='.png'):
    """
    Create MP4 video from PNG images in a directory
    
    Args:
        input_dir (str): Directory containing PNG images
        output_path (str): Output MP4 file path
        fps (int): Frames per second for the video
        image_format (str): Image file format (default: '.png')
    """
    
    # Get all image files and sort them naturally
    image_files = glob.glob(os.path.join(input_dir, f'*{image_format}'))
    image_files = natsorted(image_files)  # Natural sorting (1, 2, 10, 11...)
    
    if not image_files:
        print(f"No {image_format} files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Read first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read first image: {image_files[0]}")
        return
    
    height, width, layers = first_image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping...")
            continue
        
        # Resize image if dimensions don't match
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height))
        
        # Write frame to video
        video_writer.write(image)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Create MP4 video from PNG images')
    parser.add_argument('--input_dir', '-i', required=True, 
                       help='Directory containing PNG images')
    parser.add_argument('--output', '-o', required=True,
                       help='Output MP4 file path')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--format', default='.png',
                       help='Image file format (default: .png)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_video_from_images(args.input_dir, args.output, args.fps, args.format)

if __name__ == "__main__":
    main()
