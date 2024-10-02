import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained model
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable model CPU offloading or move to GPU if available
if device == "cuda":
    pipe.to("cuda")
else:
    pipe.enable_model_cpu_offload()

# Function to generate video based on user input
def generate_video(prompt, num_inference_steps=25, output_dir="output_videos"):
    print(f"Generating video for prompt: {prompt}")

    # Generate video frames
    video_frames_batch = pipe(prompt, num_inference_steps=num_inference_steps).frames

    # Check if frames are in the expected batch shape
    valid_frames = []
    if video_frames_batch[0].ndim == 4:  # Check if first frame has batch dimension
        # Iterate over the batch dimension
        for batch in video_frames_batch:
            for i, frame in enumerate(batch):
                # Check if the frame has a valid number of channels
                if frame.ndim == 3 and frame.shape[2] in [1, 2, 3, 4]:
                    valid_frames.append(frame)
                else:
                    print(f"Frame {i} has shape {frame.shape} and will be skipped")
    else:
        print("Expected a batch of frames, but received a different shape.")

    # Export frames to video
    if valid_frames:  # Check if any valid frames exist
        video_path = export_to_video(valid_frames)
        print(f"Video saved to {video_path}")

        # Optionally save the video to a specific directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_video_path = os.path.join(output_dir, "generated_video.mp4")
        os.rename(video_path, output_video_path)
        print(f"Video moved to {output_video_path}")
    else:
        print("No valid frames to create a video.")

# Get user input and call the function
if __name__ == "__main__":
    prompt = input("Enter a description for the video: ")
    generate_video(prompt)
