import yt_dlp as youtube_dl

def download_video(video_url, output_path):
    try:
        ydl_opts = {
            'format': 'best',
            'outtmpl': f'{output_path}/target_video.%(ext)s',
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        print("Download completed!")

    except youtube_dl.utils.DownloadError as e:
        print(f"Download error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    video_url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
    output_path = "video"
    download_video(video_url, output_path)