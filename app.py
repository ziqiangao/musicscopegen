import gradio as gr
import librosa
from PIL import Image, ImageDraw, ImageFont
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1
import io
from colorthief import ColorThief
import colorsys
import math
import os
from multiprocessing import Pool, cpu_count
import tempfile
import ffmpeg
import subprocess
import traceback
import time
import shutil

path = ""  # Update with your path

def getRenderCords(ta: list, idx: int, res: int = 1024, size: tuple = (1280, 720)) -> list:
    i = idx - res // 2
    x, y = size[0] * .9 / -2, (ta[i] - 128) * (size[1] / 2000) + (size[1] * .7 / -2)
    c = []
    while i < idx + (res // 2):
        c.append((x, y))
        i += 1
        y = (ta[i] - 128) * (size[1] / 2000) + (size[1] * .7 / -2)
        x += (size[0] * .9) / res
    return c

def center_to_top_left(coords, width=1280, height=720):
    new_coords = []
    for x, y in coords:
        new_coords.append(totopleft((x, y), width=width, height=height))
    return new_coords

def totopleft(coord, width=1280, height=720):
    return coord[0] + width / 2, height / 2 - coord[1]

def getTrigger(ad: int, a: list, max: int = 1024) -> int:
    i = ad
    while not (a[i] < 128 and not a[i + 2] < 128 or i - ad > max):
        i += 1
    return i

def extract_cover_image(mp3_file):
    audio = MP3(mp3_file, ID3=ID3)
    if audio.tags == None:
        
        return -1
    for tag in audio.tags.values():
        if isinstance(tag, APIC):
            image_data = tag.data
            cover_image = Image.open(io.BytesIO(image_data))
            return cover_image
    print("No cover image found in the MP3 file.")
    return None

def getTitleAndArtist(mp3_file):
    audio = MP3(mp3_file, ID3=ID3)
    title = audio.get('TIT2', TIT2(encoding=3, text='Unknown Title')).text[0]
    artist = audio.get('TPE1', TPE1(encoding=3, text='Unknown Artist')).text[0]
        
    
    return title, artist

def getColour(img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        img.save(tmpfile.name, format="PNG")
        color_thief = ColorThief(tmpfile.name)
        dominant_color = color_thief.get_color(quality=1)
    os.remove(tmpfile.name)
    return dominant_color

def clamp(number):
    return max(0, min(number, 1))

def normalizeColour(C) -> tuple[int, int, int]:
    cc = colorsys.rgb_to_hsv(C[0] / 255, C[1] / 255, C[2] / 255)
    ccc = colorsys.hsv_to_rgb(cc[0], clamp(1.3 * cc[1]), .8)
    return math.floor(ccc[0] * 255), math.floor(ccc[1] * 255), math.floor(ccc[2] * 255)

def normalizeColourBar(C) -> tuple[int, int, int]:
    cc = colorsys.rgb_to_hsv(C[0] / 255, C[1] / 255, C[2] / 255)
    ccc = colorsys.hsv_to_rgb(cc[0], clamp(1.4 * cc[1]), .6)
    return math.floor(ccc[0] * 255), math.floor(ccc[1] * 255), math.floor(ccc[2] * 255)

def stamp_text(draw, text, font, position, align='left'):
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x, y = position
    y -= text_height // 2
    if align == 'center':
        x -= text_width // 2
    elif align == 'right':
        x -= text_width

    draw.text((x, y), text, font=font, fill="#fff")

def linear_interpolate(start, stop, progress):
    return start + progress * (stop - start)

def filecount(p):
    files = os.listdir()
    file_count = len(files)
    return file_count

def render_frame(params):
    n, samples_array, cover_img, title, artist, dominant_color, width, height, fps, name, oscres, sr = params
    num_frames = len(samples_array) // (sr // fps)
    img = Image.new('RGB', (width, height), normalizeColour(dominant_color))
    d = ImageDraw.Draw(img)

    s = (sr // fps) * n
    if s > len(samples_array): 
        return
    e = center_to_top_left(getRenderCords(samples_array, getTrigger(s, samples_array, max=oscres),res=oscres,size=(width, height)), width=width, height=height)
    d.line(e, fill='#fff', width=2)

    cs = math.floor(min(width, height) / 2)
    cov = cover_img.resize((cs, cs))
    img.paste(cov, (((width // 2) - cs // 2), math.floor(height * .1)))

    fontT = ImageFont.truetype(path+'Lexend-Bold.ttf', 50*(min(width, height)/720)//1) 
    fontA = ImageFont.truetype(path+'Lexend-Bold.ttf', 40*(min(width, height)/720)//1) 
    fontD = ImageFont.truetype(path+'SpaceMono-Bold.ttf', 30*(min(width, height)/720)//1) 

    stamp_text(d, title, fontT, totopleft((0, min(width, height) * .3 // -2), width=width, height=height), 'center')
    stamp_text(d, artist, fontA, totopleft((0, min(width, height) * .44 // -2), width=width, height=height), 'center')

    d.line(center_to_top_left([(width * .96 // -2, height * .95 // -2), (width * .96 // 2, height * .95 // -2)], width=width, height=height),
           fill=normalizeColourBar(dominant_color), width=15 * height // 360)
    d.line(center_to_top_left([(width * .95 // -2, height * .95 // -2),
                               (linear_interpolate(width * .95 // -2, width * .95 // 2, s / len(samples_array)),
                                height * .95 // -2)],width=width, height=height), fill='#fff', width=10 * height // 360)

    
    img.save(path+f'out/{name}/{str(n)}.png', 'PNG',)

    return 1  # Indicate one frame processed

def RenderVid(af, n, fps=30):
    (ffmpeg 
     .input(path+f'out/{n}/%d.png', framerate=fps) 
     .input(af) 
     .output(n + '.mp4', vcodec='libx264', r=fps, pix_fmt='yuv420p', acodec='aac', shortest=None) 
     .run()
     )
    gr.Interface.download(f"{n}.mp4")

def main(file, name, fps=30, res: tuple=(1280,720), oscres=512, sr=11025):
    os.makedirs(path+f'out/{name}/', exist_ok=True)
    global iii
    iii = 0
    # Load the audio file
    audio_path = file
    y, sr = librosa.load(audio_path, sr=sr)  # Resample to 11025 Hz
    y_u8 = (y * 128 + 128).astype('uint8')
    samples_array = y_u8.tolist()

    # Extract cover image, title, and artist
    cover_img = extract_cover_image(audio_path)
    if cover_img is None:
        raise gr.Error("Mp3 must have a cover image")
        return  # Exit if no cover image found
    elif cover_img == -1:
        raise gr.Error("Mp3 is missing tags")
        return
        

    title, artist = getTitleAndArtist(audio_path) 
    if title == 'Unknown Title' or artist == 'Unknown Artist':
        gr.Warning('Missing Title or Artist')
    dominant_color = getColour(cover_img)

    # Frame rendering parameters
    width, height, fps = res[0], res[1], fps
    num_frames = len(samples_array) // (sr // fps)

    # Prepare parameters for each frame
    params = [(n, samples_array, cover_img, title, artist, dominant_color, width, height, fps, name, oscres, sr) for n in range(num_frames)]
    p = gr.Progress()
    try:
        with Pool(cpu_count()) as pool:
            
            num_frames = len(samples_array) // (sr // fps)
            # Use imap to get progress updates
            for _ in pool.imap_unordered(render_frame, params):
                iii += 1  # Increment frame count for progress
                p((iii,num_frames),desc="Rendering Frames")
                

    except Exception as e:
        print('Ended in error: ' + traceback.format_exc())
        gr.Info("Rendering had errored, this typically an out of range error")
    p = gr.Progress()
    p(0.5,desc="Compiling video")
    print('FFMPEG')
    ffmpeg_cmd = [
        "ffmpeg",
        '-framerate', '30',
        '-i', path+f'out/{name}/%d.png',  # Input PNG images
        '-i', f'{file}',              # Input MP3 audio
        '-c:v', 'libx264',
        '-r', '30',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-shortest', 
        '-y',
        path+f'{name}.mp4'  # Output MP4 filename
    ]
    subprocess.run(ffmpeg_cmd)

def gradio_interface(audio_file, output_name, fps=30, vidwidth=1280, vidheight=720, oscres=512, sr=11025):
    resolution = f"{vidwidth}x{vidheight}"
    res = tuple(map(int, resolution.split('x')))
    main(audio_file, output_name, fps=fps, res=res, oscres=oscres, sr=sr)
    time.sleep(5)
    
    shutil.rmtree("out")
    return f"{output_name}.mp4"

# Define Gradio interface with progress bar
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.components.File(label="Upload your MP3 file", file_count='single', file_types=['mp3']),
        gr.components.Textbox(label="Output Video Name", value='video'),
        gr.components.Slider(label="Frames per Second", minimum=20, maximum=60, step=1, value=30),
        gr.components.Slider(label="Output Video Width", minimum=100, maximum=2000, value=1280, step=2),
        gr.components.Slider(label="Output Video Height", minimum=100, maximum=2000, value=720, step=2),
        gr.components.Slider(label="Number of Visualization Segments", minimum=256, maximum=2048, step=2, value=512),
        gr.components.Slider(label="Scope Sample Rate", minimum=8000, maximum=44100, step=5, value=11025)
    ],
    outputs=gr.components.Video(label="Output"),
    title="MP3 to Video Visualization",
    description="Upload an MP3 file and configure parameters to create a visualization video."
)

# Launch Gradio interface
iface.launch()
