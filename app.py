import glob
import os
import shutil
import urllib.request
from datetime import datetime

import ffmpeg

os.environ["DLClight"] = "True"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import deeplabcut
from pathlib import Path

import streamlit as st
import os
import yaml


def main():
    download()
    st.title("ModelZoo Demo")
    model_options = deeplabcut.create_project.modelzoo.Modeloptions
    col1, col2 = st.beta_columns(2)
    with col1:
        model2use = st.selectbox('Model Choice', model_options)
        videotype = st.selectbox('Video Format', [".mp4", ".mov", ".avi", ".webm"])
        options = st.multiselect("Options", ["run_filtering", "show_plots"])
        video = st.file_uploader('Upload Video')
        if video is not None:
            filtered = False
            animal, config_path, full_video_path, videos_dir, fp = create_project(video, model2use, videotype)
            deeplabcut.analyze_videos(config_path, str(fp), videotype=videotype)
            if "run_filtering" in options:
                deeplabcut.filterpredictions(config_path, full_video_path, videotype=videotype)
                filtered = True
            deeplabcut.create_labeled_video(config_path, full_video_path, videotype=videotype, filtered=filtered)
            files = os.listdir(videos_dir)
            for file in files:
                if file.endswith("labeled" + videotype):
                    labeled_video_path = os.path.join(videos_dir, file)
                    out_video_name = animal + datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + videotype
                    out_video_path = os.path.join(os.path.dirname(labeled_video_path), out_video_name)
                    _, _ = (
                        ffmpeg.input(labeled_video_path).output(out_video_path, vcodec='h264').run(
                            overwrite_output=True))
                    st.balloons()
                    st.video(out_video_path)
    with col2:
        if "show_plots" in options and video is not None:
            deeplabcut.plot_trajectories(config_path, str(fp), videotype, filtered=filtered)
            for plot in glob.glob(os.path.join(os.path.dirname(config_path), "videos", "plot-poses", "video", "*.png")):
                st.image(plot, caption=os.path.basename(plot))


@st.cache
def download():
    urllib.request.urlretrieve(
        "https://upload.wikimedia.org/wikipedia/commons/7/7a/Cat_playing_with_a_laser_pointer.webm",
        'video.mp4')
    os.makedirs("models", exist_ok=True)
    animals = [("dog", "full_dog"), ("cat", "full_cat")]
    YourName = 'teamDLC'
    for animal in animals:
        model_dir = os.path.join("models", animal[0])
        if os.path.isdir(model_dir):
            continue
        deeplabcut.create_pretrained_project(animal[0], YourName, videos="video.mp4",
                                             videotype=".mp4",
                                             model=animal[1])
        date = datetime.today().strftime('%Y-%m-%d')
        shutil.move(animal[0] + "-" + YourName + "-" + date, model_dir)
        os.remove(os.path.join(model_dir, "videos", "video.mp4"))
    if os.path.isfile("video.mp4"):
        os.remove("video.mp4")


def create_project(video, model2use, videotype):
    animal_dict = {"full_dog": "dog", "full_cat": "cat"}
    animal = animal_dict[model2use]
    fp = Path(os.path.join("models", animal, "videos", "video" + videotype))
    shutil.rmtree(os.path.dirname(str(fp)))
    os.mkdir(os.path.dirname(str(fp)))
    fp.write_bytes(video.getvalue())
    zoo_path = Path(__file__)
    rel_path = os.path.join("..", str(fp))
    full_video_path = str((zoo_path / rel_path).resolve())
    config_path = os.path.join("models", animal, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    videos_dir = os.path.dirname(str(fp))
    cfg["project_path"] = os.path.dirname(videos_dir)
    cfg["video_sets"][str(fp)] = cfg["video_sets"].pop(
        list(cfg["video_sets"].keys())[0])
    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)
    return animal, config_path, full_video_path, videos_dir, fp


if __name__ == "__main__":
    main()
