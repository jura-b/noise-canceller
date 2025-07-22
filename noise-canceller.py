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

from livekit import rtc, api
from livekit.plugins import noise_cancellation
from dotenv import load_dotenv

SAMPLERATE = 48000
CHUNK_DURATION_MS = 10  # 10ms chunks
SAMPLES_PER_CHUNK = int(SAMPLERATE * CHUNK_DURATION_MS / 1000)
CHANNELS = 1

load_dotenv()

# Set up logger
logger = logging.getLogger("noise-canceller")

class AudioFileProcessor:
    def __init__(self, noise_filter):
        self.noise_filter = noise_filter
        self.processed_frames = []
        self.room = None

    async def process_file(self, input_path: Path, output_path: Path):
        """Process an audio file with LiveKit noise cancellation"""
        logger.info(f"Processing {input_path} -> {output_path}")
        
        # Load the input audio file
        audio_data = self._load_audio_file(input_path)
        
        # Connect to LiveKit room (required for noise cancellation authentication)
        self.room = await self._connect_to_room()
        
        try:
            # Process audio with noise cancellation
            await self._process_with_noise_cancellation(audio_data)
            
            # Save the processed audio
            self._save_output(output_path)
            logger.info(f"Processed audio saved to {output_path}")
            
        finally:
            if self.room:
                await self.room.disconnect()
                logger.info("Disconnected from LiveKit room")

    async def _process_with_noise_cancellation(self, audio_data):
        """Process audio data through noise cancellation"""
        logger.debug("Setting up noise cancellation pipeline...")
        
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
            # Similar to how recorder.py and other agents do it
            filtered_stream = rtc.AudioStream.from_participant(
                participant=self.room.local_participant,
                track_source=rtc.TrackSource.SOURCE_MICROPHONE,
                noise_cancellation=self.noise_filter
            )
            
            logger.debug("Processing audio with noise cancellation...")
            
            # Step 3: Feed audio data and capture processed output
            chunk_count = len(audio_data) // SAMPLES_PER_CHUNK
            if len(audio_data) % SAMPLES_PER_CHUNK != 0:
                chunk_count += 1
            
            # Start feeding and capturing concurrently
            feed_task = asyncio.create_task(self._feed_audio_data(file_source, audio_data, chunk_count))
            capture_task = asyncio.create_task(self._capture_filtered_audio(filtered_stream, chunk_count))
            
            # Wait for both tasks with timeout
            try:
                await asyncio.wait_for(asyncio.gather(feed_task, capture_task), timeout=120.0)
                logger.info(f"Successfully processed {len(self.processed_frames)} frames")
                
            except asyncio.TimeoutError:
                logger.warning("Processing timed out")
                feed_task.cancel()
                capture_task.cancel()
                
        except Exception as e:
            logger.error(f"Error setting up noise cancellation: {e}")
            raise
        finally:
            # Clean up resources to avoid task cancellation warnings
            if filtered_stream:
                try:
                    logger.debug("Cleaning up audio stream...")
                    await filtered_stream.aclose()
                except Exception as e:
                    logger.debug(f"Note: Audio stream cleanup completed with message: {e}")
            
            # Unpublish the track
            try:
                logger.debug("Unpublishing audio track...")
                await self.room.local_participant.unpublish_track(publication.sid)
            except Exception as e:
                logger.debug(f"Note: Track unpublish completed with message: {e}")

    async def _feed_audio_data(self, file_source, audio_data, chunk_count):
        """Feed audio data to the source with precise timing"""
        logger.debug(f"Feeding {chunk_count} audio chunks...")
        
        # Use more precise timing based on actual sample rate
        chunk_duration = SAMPLES_PER_CHUNK / SAMPLERATE  # Exact duration in seconds
        start_time = asyncio.get_event_loop().time()
        
        for i in range(chunk_count):
            start_idx = i * SAMPLES_PER_CHUNK
            end_idx = min(start_idx + SAMPLES_PER_CHUNK, len(audio_data))
            chunk = audio_data[start_idx:end_idx]
            
            # Pad last chunk if necessary with silence
            if len(chunk) < SAMPLES_PER_CHUNK:
                chunk = np.concatenate([chunk, np.zeros(SAMPLES_PER_CHUNK - len(chunk), dtype=np.int16)])
            
            # Create audio frame with proper samples count
            audio_frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SAMPLERATE,
                num_channels=CHANNELS,
                samples_per_channel=len(chunk)
            )
            
            # Feed to source
            await file_source.capture_frame(audio_frame)
            
            # More precise timing - wait for the exact moment the next chunk should be sent
            target_time = start_time + (i + 1) * chunk_duration
            current_time = asyncio.get_event_loop().time()
            delay = max(0, target_time - current_time)
            
            if delay > 0:
                await asyncio.sleep(delay)
            
            if (i + 1) % 500 == 0 or i == chunk_count - 1:
                logger.debug(f"Fed {i + 1}/{chunk_count} chunks")

    async def _capture_filtered_audio(self, filtered_stream, expected_chunks):
        """Capture the noise-cancelled audio output"""
        logger.debug("Capturing noise-cancelled audio...")
        captured = 0
        
        # Wait a moment for stream to be ready
        await asyncio.sleep(0.1)
        
        try:
            async for audio_event in filtered_stream:
                frame = audio_event.frame
                self.processed_frames.append(frame.data)
                captured += 1
                
                if captured % 500 == 0 or captured >= expected_chunks:
                    logger.debug(f"Captured {captured} processed chunks")
                
                if captured >= expected_chunks:
                    break
                    
        except Exception as e:
            logger.error(f"Error capturing processed audio: {e}")
            
        logger.info(f"Audio capture complete: {captured} chunks captured")

    def _load_audio_file(self, input_path: Path):
        """Load and preprocess audio file (supports WAV, FLAC, OGG, MP3, M4A, etc.)"""
        try:
            # Load audio file using soundfile (supports many formats)
            audio_data, sample_rate = sf.read(str(input_path), dtype='int16')
            
            # Get channels info
            if audio_data.ndim == 1:
                channels = 1
            else:
                channels = audio_data.shape[1]
            
            duration_s = len(audio_data) / sample_rate
            
            logger.info(f"Input: {sample_rate}Hz, {channels} channels, {duration_s:.2f}s")
            logger.info(f"Detected format: {input_path.suffix}")
            
            # Convert to numpy array with proper shape
            if channels == 1 and audio_data.ndim == 1:
                audio_array = audio_data
            else:
                audio_array = audio_data
            
            # Resample to 48kHz mono if needed
            if sample_rate != SAMPLERATE or channels != CHANNELS:
                audio_array = self._resample_audio(audio_array, sample_rate, channels)
                logger.info(f"Resampled to: {SAMPLERATE}Hz, {CHANNELS} channels")
            
            return audio_array
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            logger.error(f"Supported formats: WAV, FLAC, OGG, MP3 (with ffmpeg), M4A, and more")
            logger.error(f"Make sure you have ffmpeg installed for MP3/M4A support")
            raise

    def _resample_audio(self, audio_array, original_rate, original_channels):
        """High-quality resampling using LiveKit's AudioResampler (Sox library)"""
        
        # Convert to mono if stereo - simple average
        if original_channels == 2:
            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1).astype(np.int16)
            else:
                # Flat array, need to reshape
                stereo = audio_array.reshape(-1, 2)
                audio_array = stereo.mean(axis=1).astype(np.int16)
        
        # High-quality resampling with LiveKit's AudioResampler if needed
        if original_rate != SAMPLERATE:
            logger.debug(f"Resampling from {original_rate}Hz to {SAMPLERATE}Hz using LiveKit AudioResampler...")
            
            # Create LiveKit AudioResampler with high quality
            resampler = rtc.AudioResampler(
                input_rate=original_rate,
                output_rate=SAMPLERATE,
                num_channels=1,  # We've already converted to mono
                quality=rtc.AudioResamplerQuality.VERY_HIGH  # Use highest quality
            )
            
            # Create AudioFrame from the input data
            input_frame = rtc.AudioFrame(
                data=audio_array.tobytes(),
                sample_rate=original_rate,
                num_channels=1,
                samples_per_channel=len(audio_array)
            )
            
            # Push through resampler
            output_frames = resampler.push(input_frame)
            output_frames.extend(resampler.flush())  # Get any remaining samples
            
            if len(output_frames) > 0:
                # Combine all output frames
                resampled_data = b''.join(frame.data for frame in output_frames)
                audio_array = np.frombuffer(resampled_data, dtype=np.int16)
            else:
                logger.warning("Warning: No output from AudioResampler, using original data")
        
        return audio_array

    def _save_output(self, output_path: Path):
        """Save processed audio frames to output file"""
        if not self.processed_frames:
            logger.warning("Warning: No processed frames to save")
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
        logger.info(f"Connected to LiveKit Cloud room for authentication / metering")
        return room


class FileAudioSource(rtc.AudioSource):
    """Custom audio source that streams from file data"""
    def __init__(self, audio_data, sample_rate=SAMPLERATE, num_channels=CHANNELS):
        super().__init__(sample_rate, num_channels)
        self.audio_data = audio_data


def setup_logging(log_level: str):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[console_handler],
        force=True
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Process audio files with LiveKit noise cancellation (supports MP3, WAV, FLAC, OGG, M4A, AAC, and more)",
        epilog="""
Examples:
  uv run noise-canceller.py input.mp3
  uv run noise-canceller.py input.wav -o clean_audio.wav
  uv run noise-canceller.py song.flac --filter BVC
  uv run noise-canceller.py audio.m4a -o processed.wav
  
Supported formats: MP3, WAV, FLAC, OGG, M4A, AAC, AIFF, and more
Note: Some formats may require ffmpeg to be installed on your system
  
Environment variables:
  LIVEKIT_URL: Your LiveKit Cloud server URL
  LIVEKIT_API_KEY: Your LiveKit API key
  LIVEKIT_API_SECRET: Your LiveKit API secret
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input audio file (supports MP3, WAV, FLAC, OGG, M4A, AAC, etc.)"
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
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check environment
    if not os.getenv("LIVEKIT_URL"):
        logger.error("Error: LIVEKIT_URL environment variable is required")
        logger.error("Set it to your LiveKit server URL, e.g., wss://your-project.livekit.cloud")
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        logger.error(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"output/{input_path.stem}-processed.wav")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Choose noise cancellation filter
    if args.filter == "BVC":
        noise_filter = noise_cancellation.BVC()
    elif args.filter == "BVCTelephony":
        noise_filter = noise_cancellation.BVCTelephony()
    else:
        noise_filter = noise_cancellation.NC()
    
    logger.info(f"Using {args.filter if args.filter != 'NC' else 'NC (default)'} noise cancellation filter")
    logger.info(f"Output will be saved to {output_path}")
    logger.info("Note: This tool requires a LiveKit Cloud account for noise cancellation")
    
    # Process the file
    try:
        processor = AudioFileProcessor(noise_filter)
        await processor.process_file(input_path, output_path)
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())