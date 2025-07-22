#!/usr/bin/env python3

import argparse
import asyncio
import logging
import os
import sys
import wave
from pathlib import Path
import numpy as np
import soundfile as sf

# Rich imports for beautiful CLI
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

from livekit import rtc, api
from livekit.plugins import noise_cancellation
from dotenv import load_dotenv

SAMPLERATE = 48000
CHUNK_DURATION_MS = 10  # 10ms chunks
SAMPLES_PER_CHUNK = int(SAMPLERATE * CHUNK_DURATION_MS / 1000)
CHANNELS = 1

load_dotenv()

# Initialize Rich console
console = Console()

# Set up logger with Rich
logger = logging.getLogger("noise-canceller")

class AudioFileProcessor:
    def __init__(self, noise_filter):
        self.noise_filter = noise_filter
        self.processed_frames = []
        self.room = None
        self.progress = None

    async def process_file(self, input_path: Path, output_path: Path):
        """Process an audio file with LiveKit noise cancellation"""
        # Create beautiful header panel
        header = Panel.fit(
            f"🎵 [bold cyan]Audio Noise Cancellation[/bold cyan] 🎵\n"
            f"[dim]Powered by LiveKit Cloud[/dim]",
            style="cyan"
        )
        console.print(header)
        console.print()

        # Show file info table
        file_info = Table(title="📁 File Information", show_header=True, header_style="bold magenta")
        file_info.add_column("Property", style="cyan")
        file_info.add_column("Value", style="green")
        
        file_info.add_row("Input File", str(input_path))
        file_info.add_row("Output File", str(output_path))
        file_info.add_row("Filter Type", self.noise_filter.__class__.__name__)
        
        console.print(file_info)
        console.print()
        
        # Load the input audio file
        with console.status("[bold green]Loading audio file...", spinner="dots"):
            audio_data = self._load_audio_file(input_path)
        
        # Connect to LiveKit room (required for noise cancellation authentication)
        with console.status("[bold blue]Connecting to LiveKit Cloud...", spinner="dots"):
            self.room = await self._connect_to_room()
        
        try:
            # Process audio with noise cancellation and beautiful progress bars
            await self._process_with_noise_cancellation(audio_data)
            
            # Save the processed audio
            with console.status("[bold green]Saving processed audio...", spinner="dots"):
                self._save_output(output_path)
            
            # Success message
            success_panel = Panel.fit(
                f"✅ [bold green]Processing Complete![/bold green]\n"
                f"[dim]Clean audio saved to: {output_path}[/dim]",
                style="green"
            )
            console.print(success_panel)
            
        finally:
            if self.room:
                await self.room.disconnect()
                logger.debug("Disconnected from LiveKit room")

    async def _process_with_noise_cancellation(self, audio_data):
        """Process audio data through noise cancellation with progress tracking"""
        chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
        if len(audio_data) % SAMPLES_PER_CHUNK != 0:
            chunk_count += 1
        
        # Step 1: Publish the raw audio as a microphone track
        logger.debug("Publishing raw audio track...")
        file_source = FileAudioSource(audio_data, SAMPLERATE, CHANNELS)
        input_track = rtc.LocalAudioTrack.create_audio_track("raw-input", file_source)
        
        input_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await self.room.local_participant.publish_track(input_track, input_options)
        
        # Wait for track to be ready and subscribed
        await asyncio.sleep(0.5)
        
        # Step 2: Create a stream that receives from the participant with noise cancellation
        logger.debug("Setting up noise-cancelled audio stream...")
        
        filtered_stream = None
        try:
            # This is the key - create stream from participant with noise cancellation
            filtered_stream = rtc.AudioStream.from_participant(
                participant=self.room.local_participant,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                noise_cancellation=self.noise_filter
            )
            
            # Step 3: Feed audio data and capture processed output with progress bars
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                
                # Create progress tasks
                feed_task = progress.add_task("🎤 Feeding audio chunks", total=chunk_count)
                capture_task = progress.add_task("🔊 Capturing processed audio", total=chunk_count)
                
                # Start feeding and capturing concurrently
                feed_coro = self._feed_audio_data_with_progress(file_source, audio_data, chunk_count, progress, feed_task)
                capture_coro = self._capture_filtered_audio_with_progress(filtered_stream, chunk_count, progress, capture_task)
                
                # Wait for both tasks with timeout
                try:
                    await asyncio.wait_for(asyncio.gather(feed_coro, capture_coro), timeout=120.0)
                    logger.info(f"Successfully processed {len(self.processed_frames)} frames")
                    
                except asyncio.TimeoutError:
                    console.print("⚠️  [yellow]Processing timed out[/yellow]")
                    
        except Exception as e:
            console.print(f"❌ [red]Error setting up noise cancellation: {e}[/red]")
            raise
        finally:
            # Clean up resources
            if filtered_stream:
                try:
                    await filtered_stream.aclose()
                except Exception as e:
                    logger.debug(f"Audio stream cleanup completed: {e}")
            
            # Unpublish the track
            try:
                await self.room.local_participant.unpublish_track(publication.sid)
            except Exception as e:
                logger.debug(f"Track unpublish completed: {e}")

    async def _feed_audio_data_with_progress(self, file_source, audio_data, chunk_count, progress, task_id):
        """Feed audio data to the source with precise timing and progress updates"""
        chunk_duration = SAMPLES_PER_CHUNK / SAMPLERATE
        start_time = asyncio.get_event_loop().time()
        
        for i in range(chunk_count):
            start_idx = i * SAMPLES_PER_CHUNK
            end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            # Pad last chunk if necessary with silence
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.concatenate([chunk, np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16)])
            
            # Create audio frame
            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SAMPLERATE,
                num_channels=CHANNELS,
                samples_per_channel=len(chunk)
            )
            
            # Feed to source
            await file_source.capture_frame(audio_frame)
            
            # Update progress
            progress.update(task_id, advance=1)
            
            # Precise timing
            target_time = start_time + (i + 1) * chunk_duration
            current_time = asyncio.get_event_loop().time()
            delay = max(0, target_time - current_time)
            
            if delay > 0:
                await asyncio.sleep(delay)

    async def _capture_filtered_audio_with_progress(self, filtered_stream, expected_chunks, progress, task_id):
        """Capture the noise-cancelled audio output with progress updates"""
        captured = 0
        
        # Wait for stream to be ready
        await asyncio.sleep(0.1)
        
        try:
            async for audio_event in filtered_stream:
                frame = audio_event.frame
                self.processed_frames.append(frame.data)
                captured += 1
                
                # Update progress
                progress.update(task_id, advance=1)
                
                if captured >= expected_chunks:
                    break
                    
        except Exception as e:
            console.print(f"❌ [red]Error capturing processed audio: {e}[/red]")

    def _load_audio_file(self, input_path: Path):
        """Load and preprocess audio file"""
        try:
            # Load audio file using soundfile
            audio_data, sample_rate = sf.read(str(input_path), dtype='int16')
            
            # Get channels info
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
            
            duration_s = len(audio_data) / sample_rate
            
            # Create audio info table
            audio_info = Table(title="🎵 Audio Properties", show_header=True, header_style="bold blue")
            audio_info.add_column("Property", style="cyan")
            audio_info.add_column("Value", style="green")
            
            audio_info.add_row("Sample Rate", f"{sample_rate:,} Hz")
            audio_info.add_row("Channels", str(channels))
            audio_info.add_row("Duration", f"{duration_s:.2f} seconds")
            audio_info.add_row("Format", input_path.suffix.upper())
            
            console.print(audio_info)
            console.print()
            
            # Convert to numpy array with proper shape
            if channels == 1 and audio_data.ndim == 1:
                audio_array = audio_data
            else:
                audio_array = audio_data
            
            # Resample to 48kHz mono if needed
            if sample_rate != SAMPLERATE or channels != CHANNELS:
                audio_array = self._resample_audio(audio_array, sample_rate, channels)
                console.print(f"🔄 [yellow]Resampled to: {SAMPLERATE}Hz, {CHANNELS} channel(s)[/yellow]")
                console.print()
            
            return audio_array
            
        except Exception as e:
            console.print(f"❌ [red]Error loading audio file: {e}[/red]")
            console.print("[dim]Supported formats: WAV, FLAC, OGG, MP3 (with ffmpeg), M4A, and more[/dim]")
            console.print("[dim]Make sure you have ffmpeg installed for MP3/M4A support[/dim]")
            raise

    def _resample_audio(self, audio_array, original_rate, original_channels):
        """High-quality resampling using LiveKit's AudioResampler"""
        
        # Convert to mono if stereo
        if original_channels == 2:
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1).astype(np.int16)
            else:
                stereo = audio_array.reshape(-1, 2)
                audio_array = stereo.mean(axis=1).astype(np.int16)
        
        # Resample if needed
        if original_rate != SAMPLERATE:
            resampler = rtc.AudioResampler(
                input_rate=original_rate,
                output_rate=SAMPLERATE,
                num_channels=1,
                quality=rtc.AudioResamplerQuality.VERY_HIGH
            )
            
            input_frame = rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=original_rate,
                num_channels=1,
                samples_per_channel=len(audio_array)
            )
            
            output_frames = resampler.push(input_frame)
            output_frames.extend(resampler.flush())
            
            if len(output_frames) > 0:
                resampled_data = b''.join(frame.data for frame in output_frames)
                audio_array = np.frombuffer(resampled_data, dtype=np.int16)
            else:
                console.print("⚠️  [yellow]Warning: No output from AudioResampler, using original data[/yellow]")
        
        return audio_array

    def _save_output(self, output_path: Path):
        """Save processed audio frames to output file"""
        if not self.processed_frames:
            console.print("⚠️  [yellow]Warning: No processed frames to save[/yellow]")
            return
            
        with wave.open(str(output_path), 'wb') as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLERATE)
            
            for frame_data in self.processed_frames:
                wav_file.writeframes(frame_data)

    async def _connect_to_room(self):
        """Connect to LiveKit Cloud room for authentication / metering"""
        token = (
            api.AccessToken()
            .with_identity("noise-canceller")
            .with_name("Noise Canceller")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room="noise-canceller-room",
                    agent=True,
                )
            )
            .to_jwt()
        )
        
        url = os.getenv("LIVEKIT_URL")
        if not url:
            raise ValueError("LIVEKIT_URL environment variable is required")

        room = rtc.Room()
        await room.connect(
            url,
            token,
            options=rtc.RoomOptions(
                auto_subscribe=False,
            ),
        )
        logger.debug("Connected to LiveKit Cloud room for authentication / metering")
        return room


class FileAudioSource(rtc.AudioSource):
    """Custom audio source that streams from file data"""
    def __init__(self, audio_data, sample_rate=SAMPLERATE, num_channels=CHANNELS):
        super().__init__(sample_rate, num_channels)
        self.audio_data = audio_data


def setup_logging(log_level: str):
    """Setup beautiful Rich logging configuration"""
    level = getattr(logging, log_level.upper())
    
    # Create Rich handler with beautiful formatting
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_suppress=[rtc, api, noise_cancellation]
    )
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[rich_handler],
        force=True
    )


async def main():
    parser = argparse.ArgumentParser(
        description="🎵 Process audio files with LiveKit noise cancellation",
        epilog="""
✨ Examples:
  uv run noise-canceller.py input.mp3
  uv run noise-canceller.py input.wav -o clean_audio.wav
  uv run noise-canceller.py song.flac --filter BVC
  uv run noise-canceller.py audio.m4a -o processed.wav
  
📁 Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, AIFF, and more
📝 Note: Some formats may require ffmpeg to be installed
  
🔧 Environment variables:
  LIVEKIT_URL: Your LiveKit Cloud server URL
  LIVEKIT_API_KEY: Your LiveKit API key  
  LIVEKIT_API_SECRET: Your LiveKit API secret
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: output/<input-file-name>-processed.wav)"
    )
    parser.add_argument(
        "--filter",
        choices=["NC", "BVC", "BVCTelephony"],
        default="NC",
        help="Noise cancellation filter type (default: NC)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup beautiful logging
    setup_logging(args.log_level)
    
    # Check environment with beautiful error messages
    if not os.getenv("LIVEKIT_URL"):
        error_panel = Panel.fit(
            "❌ [bold red]Missing Environment Variable[/bold red]\n\n"
            "[dim]LIVEKIT_URL environment variable is required.[/dim]\n"
            "[dim]Set it to your LiveKit server URL, e.g.:[/dim]\n"
            "[cyan]export LIVEKIT_URL=wss://your-project.livekit.cloud[/cyan]",
            style="red"
        )
        console.print(error_panel)
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        console.print(f"❌ [red]Input file '{input_path}' does not exist[/red]")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"output/{input_path.stem}-processed.wav")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose noise cancellation filter
    filter_map = {
        "BVC": noise_cancellation.BVC(),
        "BVCTelephony": noise_cancellation.BVCTelephony(),
        "NC": noise_cancellation.NC()
    }
    noise_filter = filter_map[args.filter]
    
    # Process the file
    try:
        processor = AudioFileProcessor(noise_filter)
        await processor.process_file(input_path, output_path)
        
        # Final success message
        final_panel = Panel.fit(
            "🎉 [bold green]All Done![/bold green]\n"
            f"[dim]Your noise-cancelled audio is ready at:[/dim]\n"
            f"[cyan]{output_path}[/cyan]",
            style="green"
        )
        console.print()
        console.print(final_panel)
        
    except Exception as e:
        error_panel = Panel.fit(
            f"💥 [bold red]Processing Failed[/bold red]\n\n"
            f"[dim]Error details:[/dim]\n"
            f"[red]{str(e)}[/red]",
            style="red"
        )
        console.print(error_panel)
        
        if args.log_level == "DEBUG":
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())