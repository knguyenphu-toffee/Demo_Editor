import os
import shutil
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import median_filter
import argparse
from tqdm import tqdm
import subprocess
import sys
import matplotlib.pyplot as plt

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Apply a low-pass filter to smooth the audio envelope"""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def detect_audio_segments(audio_path, threshold_db=-40, min_silence_duration=1.5, sample_rate=22050, debug=False):
    """
    Detect segments with audio activity with improved detection logic
    
    Args:
        audio_path: Path to audio file
        threshold_db: Volume threshold in dB below which audio is considered silence
        min_silence_duration: Minimum duration of silence to consider (in seconds)
        sample_rate: Sample rate for audio processing
        debug: Show debug plots and info
    
    Returns:
        List of tuples (start_time, end_time) for segments with audio
    """
    print(f"   • Loading audio file: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    print(f"   • Audio loaded: {len(y)/sr:.1f}s duration, {sr}Hz sample rate")
    
    # Calculate RMS energy in overlapping frames
    hop_length = int(sr * 0.01)  # 10ms hop
    frame_length = int(sr * 0.05)  # 50ms frame (longer for better detection)
    
    # Get RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB, handle zeros
    rms_clipped = np.maximum(rms, 1e-8)  # Prevent log(0)
    rms_db = 20 * np.log10(rms_clipped)
    
    # Apply smoothing to reduce noise
    rms_smooth = median_filter(rms_db, size=9)
    
    # More sophisticated threshold determination
    # Remove the very quiet parts for statistics
    non_silent_mask = rms_smooth > -80
    if np.sum(non_silent_mask) > 0:
        non_silent_db = rms_smooth[non_silent_mask]
        
        # Calculate various statistics
        percentile_10 = np.percentile(non_silent_db, 10)
        percentile_25 = np.percentile(non_silent_db, 25)
        percentile_50 = np.percentile(non_silent_db, 50)
        percentile_75 = np.percentile(non_silent_db, 75)
        mean_db = np.mean(non_silent_db)
        std_db = np.std(non_silent_db)
        
        print(f"   • Audio level statistics:")
        print(f"     - 10th percentile: {percentile_10:.1f} dB")
        print(f"     - 25th percentile: {percentile_25:.1f} dB")
        print(f"     - 50th percentile: {percentile_50:.1f} dB")
        print(f"     - 75th percentile: {percentile_75:.1f} dB")
        print(f"     - Mean: {mean_db:.1f} dB")
        print(f"     - Std: {std_db:.1f} dB")
        
        # Adaptive threshold calculation
        # Use a combination of percentile-based and statistics-based thresholds
        adaptive_threshold = min(
            percentile_25 - 3,  # 3dB below 25th percentile
            mean_db - std_db * 1.5,  # 1.5 standard deviations below mean
            percentile_10 + 5   # 5dB above 10th percentile
        )
        
        # Use the more conservative threshold
        effective_threshold = max(threshold_db, adaptive_threshold)
        
        print(f"   • Threshold analysis:")
        print(f"     - User threshold: {threshold_db:.1f} dB")
        print(f"     - Adaptive threshold: {adaptive_threshold:.1f} dB")
        print(f"     - Effective threshold: {effective_threshold:.1f} dB")
    else:
        effective_threshold = threshold_db
        print(f"   • Warning: No audio detected above -80dB, using user threshold: {effective_threshold:.1f} dB")
    
    # Find segments above threshold
    above_threshold = rms_smooth > effective_threshold
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop_length)
    
    # Debug visualization
    if debug:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(times, rms_db, alpha=0.5, label='Raw RMS (dB)')
        plt.plot(times, rms_smooth, label='Smoothed RMS (dB)')
        plt.axhline(y=effective_threshold, color='r', linestyle='--', label=f'Threshold ({effective_threshold:.1f} dB)')
        plt.ylabel('dB')
        plt.title('Audio Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        plt.plot(times, above_threshold.astype(int), label='Above Threshold')
        plt.ylabel('Audio Detected')
        plt.title('Threshold Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Show histogram of audio levels
        plt.subplot(3, 1, 3)
        plt.hist(rms_smooth, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=effective_threshold, color='r', linestyle='--', label=f'Threshold ({effective_threshold:.1f} dB)')
        plt.xlabel('dB')
        plt.ylabel('Count')
        plt.title('Distribution of Audio Levels')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Find continuous segments with improved logic
    segments = []
    in_segment = False
    start_time = 0
    min_audio_duration = 0.1  # Minimum 100ms for audio segment
    
    # Use hysteresis to avoid flickering
    hysteresis_frames = int(0.2 * sr / hop_length)  # 200ms hysteresis
    
    for i, (t, is_audio) in enumerate(zip(times, above_threshold)):
        if is_audio and not in_segment:
            # Look ahead to confirm this is sustained audio
            look_ahead = min(hysteresis_frames, len(above_threshold) - i)
            if look_ahead > 0:
                future_audio_ratio = np.sum(above_threshold[i:i+look_ahead]) / look_ahead
                if future_audio_ratio > 0.3:  # 30% of next frames should be audio
                    start_time = t
                    in_segment = True
        elif not is_audio and in_segment:
            # Look ahead to confirm this is sustained silence
            look_ahead = min(hysteresis_frames, len(above_threshold) - i)
            if look_ahead > 0:
                future_audio_ratio = np.sum(above_threshold[i:i+look_ahead]) / look_ahead
                if future_audio_ratio < 0.2:  # Less than 20% of next frames should be audio
                    if t - start_time >= min_audio_duration:
                        segments.append((start_time, t))
                    in_segment = False
            else:
                # End of file
                if t - start_time >= min_audio_duration:
                    segments.append((start_time, t))
                in_segment = False
    
    # Handle last segment
    if in_segment and times[-1] - start_time >= min_audio_duration:
        segments.append((start_time, times[-1]))
    
    # Merge nearby segments that are too close
    merged_segments = []
    if segments:
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if start - current_end < min_silence_duration:
                # Merge segments
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_segments.append((current_start, current_end))
    
    # Filter out very short segments
    final_segments = [(s, e) for s, e in merged_segments if e - s >= min_audio_duration]
    
    # Debug output
    print(f"   • Audio segments found: {len(final_segments)}")
    total_audio_time = sum(e - s for s, e in final_segments)
    total_time = times[-1] if len(times) > 0 else 0
    audio_percentage = (total_audio_time / total_time * 100) if total_time > 0 else 0
    
    print(f"   • Total audio time: {total_audio_time:.1f}s ({audio_percentage:.1f}% of video)")
    print(f"   • Total silent time: {total_time - total_audio_time:.1f}s ({100 - audio_percentage:.1f}% of video)")
    
    for i, (start, end) in enumerate(final_segments):
        print(f"     {i+1}. {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
    
    return final_segments

def process_video_simple(input_path, output_path, segments_to_process, temp_dir):
    """
    Process video by cutting into segments, speeding up silent ones, then concatenating
    """
    print(f"\n🎬 Stage 7: Cutting video into {len(segments_to_process)} segments...")
    
    # Step 1: Cut the video into segments
    segment_files = []
    
    print("   Step 1: Cutting original video into segments...")
    for i, segment in enumerate(tqdm(segments_to_process, desc="   Cutting")):
        temp_filename = os.path.join(temp_dir, f"segment_{i:03d}_cut.mp4")
        
        # Re-encode during cut to ensure proper keyframes
        cut_cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', input_path,
            '-ss', str(segment['start']),
            '-t', str(segment['end'] - segment['start']),
            '-c:v', 'libx264', '-preset', 'fast',  # Re-encode video
            '-c:a', 'aac', '-b:a', '192k',        # Re-encode audio
            '-avoid_negative_ts', 'make_zero',     # Fix timestamp issues
            '-fflags', '+genpts',                  # Generate presentation timestamps
            temp_filename
        ]
        
        try:
            subprocess.run(cut_cmd, check=True, capture_output=True)
            segment_files.append({
                'filename': temp_filename,
                'type': segment['type'],
                'speed': segment['speed'],
                'index': i
            })
        except subprocess.CalledProcessError as e:
            print(f"\n   ❌ Error cutting segment {i+1}: {e.stderr.decode()}")
            return False, []
    
    print(f"   ✓ Successfully cut {len(segment_files)} segments")
    
    # Step 2: Speed up the silent segments
    print("\n   Step 2: Speeding up silent segments...")
    final_segments = []
    
    for seg in tqdm(segment_files, desc="   Processing"):
        if seg['type'] == 'silent' and seg['speed'] > 1.0:
            # Speed up this segment
            output_filename = os.path.join(temp_dir, f"segment_{seg['index']:03d}_final.mp4")
            
            speed = seg['speed']
            video_filter = f"setpts={1/speed}*PTS"
            
            # Handle audio speed (atempo has max 2.0)
            if speed <= 2.0:
                audio_filter = f"atempo={speed}"
            elif speed <= 4.0:
                audio_filter = f"atempo=2.0,atempo={speed/2.0}"
            else:
                audio_filter = "atempo=2.0,atempo=2.0"
            
            speed_cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-i', seg['filename'],
                '-vf', video_filter,
                '-af', audio_filter,
                '-c:v', 'libx264', '-preset', 'medium',
                '-c:a', 'aac', '-b:a', '192k',
                output_filename
            ]
            
            try:
                subprocess.run(speed_cmd, check=True, capture_output=True)
                final_segments.append(output_filename)
                # Remove the original cut file
                os.remove(seg['filename'])
            except subprocess.CalledProcessError as e:
                print(f"\n   ❌ Error speeding up segment {seg['index']+1}: {e.stderr.decode()}")
                # Use original if speed-up fails
                final_segments.append(seg['filename'])
        else:
            # Audio segment - keep as is but rename for consistency
            output_filename = os.path.join(temp_dir, f"segment_{seg['index']:03d}_final.mp4")
            # Instead of rename, we'll copy to ensure compatibility
            shutil.copy2(seg['filename'], output_filename)
            os.remove(seg['filename'])
            final_segments.append(output_filename)
    
    print(f"   ✓ Processed all segments")
    
    # Step 3: Concatenate all segments
    print("\n   Step 3: Concatenating all segments back together...")
    concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
    
    # Create concat list
    with open(concat_list_path, 'w') as f:
        for segment_file in final_segments:
            f.write(f"file '{os.path.basename(segment_file)}'\n")
    
    # Concatenate using ffmpeg with re-encoding for smooth playback
    concat_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c:v', 'libx264', '-preset', 'medium',  # Re-encode for consistency
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',  # Optimize for streaming
        output_path
    ]
    
    # Change to temp directory for concat to work
    original_dir = os.getcwd()
    os.chdir(temp_dir)
    
    try:
        subprocess.run(concat_cmd, check=True, capture_output=True)
        os.chdir(original_dir)
        print(f"   ✓ Successfully concatenated all segments")
        return True, final_segments + [concat_list_path]
    except subprocess.CalledProcessError as e:
        os.chdir(original_dir)
        print(f"   ❌ Error concatenating: {e.stderr.decode()}")
        return False, final_segments + [concat_list_path]

def process_video(input_path, output_path, speed_silent=2.5, speed_audio=1.0, 
                 threshold_db=-40, min_silence_duration=1.5, debug=False):
    """
    Process video to speed up silent parts
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        speed_silent: Speed multiplier for silent parts
        speed_audio: Speed multiplier for audio parts (usually 1.0)
        threshold_db: Audio threshold in dB
        min_silence_duration: Minimum silence duration in seconds
        debug: Show debug visualization
    """
    print(f"\n📂 Stage 3: Loading video for analysis...")
    
    # Get video info using ffprobe
    probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                 'format=duration:stream=width,height,r_frame_rate', 
                 '-of', 'json', input_path]
    
    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        import json
        probe_data = json.loads(result.stdout)
        duration = float(probe_data['format']['duration'])
        print(f"   ✓ Video loaded successfully")
        print(f"   ✓ Duration: {duration:.1f} seconds")
    except:
        # Fallback to moviepy for getting duration
        video = VideoFileClip(input_path)
        duration = video.duration
        print(f"   ✓ Video loaded successfully")
        print(f"   ✓ Duration: {duration:.1f} seconds")
        video.close()
    
    # Create temp directory
    script_dir = os.path.dirname(os.path.abspath(input_path))
    temp_dir = os.path.join(script_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Extract audio temporarily
    temp_audio = os.path.join(temp_dir, "temp_audio.wav")
    print(f"\n🎵 Stage 4: Extracting audio track...")
    print(f"   • Saving to temp directory: {temp_dir}")
    
    # Use ffmpeg to extract audio
    audio_cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', input_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '22050', '-ac', '1',
        temp_audio
    ]
    
    try:
        subprocess.run(audio_cmd, check=True)
        print(f"   ✓ Audio extracted successfully")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to extract audio: {e}")
        return
    
    # Detect audio segments
    print(f"\n🔊 Stage 5: Analyzing audio for silence detection...")
    print(f"   • Threshold: {threshold_db} dB")
    print(f"   • Minimum silence duration: {min_silence_duration}s")
    audio_segments = detect_audio_segments(temp_audio, threshold_db, min_silence_duration, debug=debug)
    
    # Check if we found any segments
    if not audio_segments:
        print(f"   ⚠️  No audio segments detected!")
        print(f"   • This might mean the entire video is silent, or the threshold is too high")
        print(f"   • Try lowering the threshold (e.g., -50 dB or -60 dB)")
        print(f"   • Use --debug flag to see audio level visualization")
        return
    
    # Create timeline showing what will happen to each segment
    print(f"\n✂️  Stage 6: Creating segment timeline...")
    segments_to_process = []
    current_pos = 0
    
    # Add padding to audio segments (0.1 seconds before and after)
    audio_padding = 0.1
    
    for i, (audio_start, audio_end) in enumerate(audio_segments):
        # Adjust audio segment boundaries with padding
        padded_start = max(0, audio_start - audio_padding)
        padded_end = min(duration, audio_end + audio_padding)
        
        # Silent segment before audio (if any)
        if current_pos < padded_start:
            segments_to_process.append({
                'start': current_pos,
                'end': padded_start,
                'speed': speed_silent,
                'type': 'silent'
            })
            print(f"   • Silent: {current_pos:.1f}s - {padded_start:.1f}s (speed: {speed_silent}x)")
        
        # Audio segment with padding
        segments_to_process.append({
            'start': padded_start,
            'end': padded_end,
            'speed': speed_audio,
            'type': 'audio'
        })
        print(f"   • Audio:  {padded_start:.1f}s - {padded_end:.1f}s (speed: {speed_audio}x) [padded ±{audio_padding}s]")
        
        current_pos = padded_end
    
    # Final silent segment
    if current_pos < duration:
        segments_to_process.append({
            'start': current_pos,
            'end': duration,
            'speed': speed_silent,
            'type': 'silent'
        })
        print(f"   • Silent: {current_pos:.1f}s - {duration:.1f}s (speed: {speed_silent}x)")
    
    # Check if we have any silent segments to speed up
    silent_segments = [s for s in segments_to_process if s['type'] == 'silent']
    if not silent_segments:
        print(f"   ⚠️  No silent segments found to speed up!")
        print(f"   • The entire video appears to have audio")
        print(f"   • Consider increasing the minimum silence duration or lowering the threshold")
        return
    
    print(f"   ✓ Found {len(silent_segments)} silent segments to speed up")
    
    # Process using simplified 3-step approach
    success, temp_files = process_video_simple(input_path, output_path, segments_to_process, temp_dir)
    
    if success:
        # Calculate statistics
        original_duration = duration
        expected_duration = sum([(s['end'] - s['start']) / s['speed'] for s in segments_to_process])
        
        # Get actual duration of output
        probe_output = ['ffprobe', '-v', 'error', '-show_entries', 
                       'format=duration', '-of', 'json', output_path]
        try:
            result = subprocess.run(probe_output, capture_output=True, text=True, check=True)
            import json
            actual_duration = float(json.loads(result.stdout)['format']['duration'])
        except:
            actual_duration = expected_duration
        
        time_saved = original_duration - actual_duration
        
        print(f"\n📊 Stage 9: Statistics")
        print(f"   • Original duration: {original_duration:.1f}s")
        print(f"   • Expected new duration: {expected_duration:.1f}s")
        print(f"   • Actual new duration: {actual_duration:.1f}s")
        print(f"   • Time saved: {time_saved:.1f}s ({time_saved/original_duration*100:.1f}%)")
    
    # Cleanup
    print(f"\n🧹 Stage 10: Cleaning up temporary files...")
    try:
        # Remove temp audio
        if os.path.exists(temp_audio):
            os.remove(temp_audio)
        
        # Remove all temp files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Remove temp directory if empty
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            print(f"   ✓ Removed temp directory")
        else:
            print(f"   ✓ Temp files cleaned")
            
    except Exception as e:
        print(f"   ⚠️  Cleanup warning: {e}")
    
    print(f"   ✓ Cleanup complete")

def find_video_file(directory='.'):
    """Find video files in the directory (excluding edited ones)"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    print("🔍 Stage 1: Looking for video files...")
    print(f"   Searching in: {os.path.abspath(directory)}")
    
    video_files = []
    try:
        files = os.listdir(directory)
        print(f"   Found {len(files)} files in directory")
        
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                # Skip already edited files
                if '_edited' not in file.lower():
                    video_files.append(file)
                    print(f"   ✓ Found video: {file}")
                else:
                    print(f"   ⏭️  Skipping edited video: {file}")
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return None
    
    if not video_files:
        print("❌ No video files found in the current directory!")
        print("   Make sure your video file:")
        print("   • Has a supported extension (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm)")
        print("   • Doesn't contain '_edited' in the filename")
        print(f"   • Is in the same directory as this script: {os.path.abspath(directory)}")
        return None
    elif len(video_files) == 1:
        print(f"✅ Selected video: {video_files[0]}")
        return video_files[0]
    else:
        print(f"⚠️  Found multiple video files:")
        for i, file in enumerate(video_files, 1):
            print(f"   {i}. {file}")
        return video_files

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except:
        return False

def main():
    # Check for ffmpeg
    if not check_ffmpeg():
        print("❌ Error: ffmpeg is not installed or not in PATH")
        print("Please install ffmpeg:")
        print("  • Mac: brew install ffmpeg")
        print("  • Windows: Download from ffmpeg.org")
        print("  • Linux: sudo apt-get install ffmpeg")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Speed up silent parts of a video')
    parser.add_argument('input', nargs='?', help='Input video file path', default=None)
    parser.add_argument('-o', '--output', help='Output video file path', default=None)
    parser.add_argument('-s', '--speed-silent', type=float, default=2.5,
                       help='Speed multiplier for silent parts (default: 2.5)')
    parser.add_argument('-a', '--speed-audio', type=float, default=1.0,
                       help='Speed multiplier for audio parts (default: 1.0)')
    parser.add_argument('-t', '--threshold', type=float, default=-40,
                       help='Audio threshold in dB (default: -40)')
    parser.add_argument('-d', '--min-silence', type=float, default=1.5,
                       help='Minimum silence duration in seconds (default: 1.5)')
    parser.add_argument('-f', '--output-folder', help='Output folder name', default='edited_videos')
    parser.add_argument('--debug', action='store_true', help='Show debug visualization of audio levels')
    
    args = parser.parse_args()
    
    # Auto-detect video if no input provided
    if args.input is None:
        # Get the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        found = find_video_file(script_dir)
        if found is None:
            return
        elif isinstance(found, list):
            # Multiple videos found, ask user to specify
            print("\nPlease specify which video to process by running:")
            print(f"python {os.path.basename(__file__)} <video_filename>")
            return
        else:
            # Build full path to the video file
            args.input = os.path.join(script_dir, found)
    
    # Generate output path if not provided
    if args.output is None:
        # Get the directory of the input file
        input_dir = os.path.dirname(args.input)
        if not input_dir:
            input_dir = '.'
        
        # Create output folder
        output_dir = os.path.join(input_dir, args.output_folder)
        print(f"\n📁 Stage 2: Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename and replace _raw with _edited
        filename = os.path.basename(args.input)
        base, ext = os.path.splitext(filename)
        
        # Replace _raw with _edited, or just add _edited if _raw not present
        if '_raw' in base:
            output_base = base.replace('_raw', '_edited')
        else:
            output_base = f"{base}_edited"
        
        output_filename = f"{output_base}{ext}"
        
        # Full output path
        args.output = os.path.join(output_dir, output_filename)
        print(f"📹 Output will be saved as: {args.output}")
    
    print("\n" + "="*60)
    print("🚀 Starting video processing...")
    print("="*60)
    
    # Process video
    process_video(
        args.input,
        args.output,
        speed_silent=args.speed_silent,
        speed_audio=args.speed_audio,
        threshold_db=args.threshold,
        min_silence_duration=args.min_silence,
        debug=args.debug
    )
    
    print("\n" + "="*60)
    print("✨ Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()