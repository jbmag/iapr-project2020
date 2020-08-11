from project_functions import extract_frames, get_operands, track_arrow, generate_video
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Compute equation from video.')
    parser.add_argument('--input', required=True, help='Path to the initial video.')
    parser.add_argument('--output', required=True, help='Path to the output video.')
    args = parser.parse_args()

    im = extract_frames(args.input)
    print(im.shape)

    # extracts digits and operands from frames
    all_objects, centers = get_operands(im, avoid_shaky_plus=True)
    arrow_boxes = track_arrow(im)
    # centers[:, [0, 1]] = centers[:, [1, 0]]  # Do it in the code
    characters = generate_video(images=im, centers=centers, all_objects=all_objects, 
                                arrow_boxes=arrow_boxes, video_output_path=args.output)