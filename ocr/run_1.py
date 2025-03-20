import argparse
import json
import os
import videodb
import yaml

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from utils import create_directories, setup_logging, save_summary
from tasks import get_task


def load_yaml_config(file_path: str) -> dict:
    yaml_path = Path(file_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "Vision Language Models Benchmarking", add_help=False
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        type=str,
        nargs="+",
        choices=[
            "benchmark",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-2.0-flash",
            "claude-3-5-sonnet-latest",
            "easyocr",
            "rapidocr",
            'mistral'
        ],
    )

    parser.add_argument("--num_vids", default=100, type=int)

    parser.add_argument("--vid_index", default=None, type=int)

    parser.add_argument(
        "--generate_images",
        action="store_true",
        help="Generate images with original image, predicted text, and ground truth text"
    )

    parser.add_argument("--download_percent", default=20, type=float, help="Percentage of images to download (0-100)")

    return parser

def download_image(image_url, download_path):
    import requests
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(download_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return download_path
        else:
            print(f"Failed to download image: {image_url}")
            return None
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None

from PIL import Image, ImageDraw, ImageFont

def generate_text_image(image_path, predicted_text, ground_truth_text, output_image_path):

    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font_size = 20

        # Load a font
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        margin = 10
        background_color = (0, 0, 0, 128)

        def draw_text(draw, position, text, font, background_color):
            # Get the bounding box of the text
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate background position
            bg_pos = [position[0] - margin, position[1] - margin,
                      position[0] + text_width + margin, position[1] + text_height + margin]

            # Draw background rectangle
            draw.rectangle(bg_pos, fill=background_color)

            # Draw text
            draw.text(position, text, font=font, fill=(255, 255, 255))

        # Positions for predicted and ground truth texts
        pred_pos = (margin, image.height - 2 * font_size - 3 * margin)
        gt_pos = (margin, image.height - font_size - 2 * margin)

        # Draw the texts
        draw_text(draw, pred_pos, f"Predicted: {predicted_text}", font, background_color)
        draw_text(draw, gt_pos, f"Ground Truth: {ground_truth_text}", font, background_color)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

        # Save the annotated image
        image.save(output_image_path)

    except Exception as e:
        print(f"Error generating annotated image: {e}")

def main(args):
    # get the task name and config
    task, config = get_task("ocr")

    # setup directories to store the result
    args.openai_results_dir = config.OPENAI_RESULTS_DIR
    # args.anthropic_results_dir = config.ANTHROPIC_RESULTS_DIR
    args.google_results_dir = config.GOOGLE_RESULTS_DIR
    args.ocr_results_dir = config.OCR_RESULTS_DIR
    args.mistral_results_dir = config.MISTRAL_RESULTS_DIR

    args.openai_evaluation_dir = config.OPENAI_EVALUATION_DIR
    # args.anthropic_evaluation_dir = config.ANTHROPIC_EVALUATION_DIR
    args.google_evaluation_dir = config.GOOGLE_EVALUATION_DIR
    args.ocr_evaluation_dir = config.OCR_EVALUATION_DIR
    args.mistral_evaluation_dir = config.MISTRAL_EVALUATION_DIR

    args.save_paths = create_directories(args)

    current_run = f"ocr_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    # get prompt
    yaml_file = load_yaml_config("prompts.yaml")
    prompt = yaml_file["ocr"]

    # get the task processor
    processor = task(prompt)

    # establish VideoDB connection and get the data
    conn = processor.establish_videodb_connection()

    # get videos
    try:
        if config.VIDEO_IDS:
            video_ids_list = list(config.VIDEO_IDS.values())  # Convert values to a list
            if args.vid_index is not None:
                video_ids_list = [video_ids_list[args.vid_index]]
            videos = processor.get_videos(
                conn=conn,
                video_ids=video_ids_list,
                collection_id=config.COLLECTION_ID,
                num_vids=args.num_vids,
            )
        else:
            videos = processor.get_videos(
                conn=conn, collection_id=config.COLLECTION_ID, num_vids=args.num_vids
            )
    except videodb.exceptions.AuthenticationError:
        print(
            "Please make sure VIDEO_DB_API_KEY is set in your .env like VIDEO_DB_API_KEY=sk-****-****"
        )
        return
    except Exception as e:
        print(f"Run failed due to {e}")
        return

    # itereate through all the models
    for path in args.save_paths:
        model_name = os.path.basename(path)

        logger, current_run_dir = setup_logging(path, current_run)

        logger.info(
            f"################################ Running {model_name} Model on OCR Prompt ################################\n"
        )

        # iterate through all the videos
        for video in tqdm(videos, desc="Processing videos", unit="video"):

            video_scenes = processor.get_scenes(video)
            outputs = processor.run(model_name, video_scenes, video.id)

            if outputs is not None:
                json_file = os.path.join(current_run_dir, f"{video.id}_output.json")
                with open(json_file, "w") as file:
                    json.dump(outputs, file)
                logger.info(f"model results of {video.id} saved to {json_file}")

            else:
                logger.info(f"failed to save model results of {video.id}")

            # Evaluation

            # load ground truth
            gt_file = os.path.join(
                config.OCR_GROUND_TRUTH_DIR, f"{video.id}_ground_truth.json"
            )
            with open(gt_file, "r", encoding='utf-8') as file:
                video_ground_truth = json.load(file)

            video_result = processor.evaluate(outputs, video_ground_truth)

            # save it in evaluation directory
            os.makedirs(os.path.join(current_run_dir, "evaluations"), exist_ok=True)

            eval_json_file = os.path.join(
                current_run_dir, "evaluations", f"{video.id}.json"
            )
            with open(eval_json_file, "w") as file:
                json.dump(video_result, file)

            logger.info(f"results evaluations of {video.id} saved to {eval_json_file}")

            import random

            if args.generate_images:
                for i, scene in enumerate(video_scenes):
                    if random.uniform(0, 100) <= args.download_percent:
                        image_download_path = os.path.join(current_run_dir, "original_images", f"{scene.id}.png")
                        os.makedirs(os.path.dirname(image_download_path), exist_ok=True)
                        downloaded_image = download_image(outputs[i]['image'][0], image_download_path)
                        # downloaded_image = download_image(video_result[i]['image'], image_download_path)
                        
                        if downloaded_image:
                            output_img_path = os.path.join(current_run_dir, "annotated_images", f"{scene.id}_annotated.png")
                            generate_text_image(downloaded_image, video_result[i]['ocr'], video_result[i]['ground_truth'], output_img_path)


    # Evaluation summary
    save_summary(current_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Vision Languange Models Benchmarking", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    main(args)
