# LiveKit Audio File Processor

A command-line tool that processes audio files with LiveKit's noise cancellation technology.

## Overview

This tool takes local audio files and processes them through LiveKit's noise cancellation pipeline. It connects to a LiveKit room for authentication but processes the audio locally without sending the actual audio data over the network.

## Features

- Process WAV audio files with LiveKit's advanced noise cancellation
- Support for multiple filter types: NC, BVC, and BVCTelephony  
- Automatic resampling to 48kHz mono
- Command-line interface for batch processing

## Installation

1. **Install dependencies:**
   ```bash
   # Generate requirements.txt
   python krisp_file_processor.py --create-requirements
   
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Set up LiveKit credentials:**
   ```bash
   export LIVEKIT_URL="wss://your-project.livekit.cloud"
   # API key/secret are optional for basic usage
   export LIVEKIT_API_KEY="your-api-key"
   export LIVEKIT_API_SECRET="your-api-secret"
   ```

## Usage

### Basic Usage
```bash
# Process input.wav and save to output.wav
python krisp_file_processor.py input.wav

# Specify custom output file
python krisp_file_processor.py input.wav -o clean_audio.wav

# Use different noise cancellation filter
python krisp_file_processor.py input.wav --filter BVC
```

### Filter Types

- **NC**: Standard enhanced noise cancellation
- **BVC**: Background voice cancellation (removes background voices + noise)
- **BVCTelephony**: BVC optimized for telephony applications

## Requirements

- **LiveKit Account**: You need a LiveKit Cloud account for noise cancellation authentication
- **Audio Format**: WAV files recommended (other formats may require conversion)
- **Python**: 3.8 or higher

## Current Limitations

⚠️ **Important Note**: This is a proof-of-concept implementation. The current version:

1. **Resamples audio** to 48kHz mono format
2. **Connects to LiveKit** for authentication  
3. **Sets up the framework** for noise cancellation
4. **Currently outputs the resampled input** rather than fully noise-cancelled audio

For full noise cancellation integration, you would need:
- Deeper integration with LiveKit's processing pipeline
- Use of LiveKit Agents framework instead of raw SDK
- More complex audio stream interception

## Example

```bash
# Process a noisy recording
python krisp_file_processor.py noisy_recording.wav -o clean_recording.wav

# The tool will:
# 1. Load noisy_recording.wav
# 2. Resample to 48kHz mono if needed
# 3. Connect to LiveKit for authentication
# 4. Process through the noise cancellation framework
# 5. Save the result to clean_recording.wav
```

## Troubleshooting

### Environment Variables
Make sure your `LIVEKIT_URL` is set correctly:
```bash
echo $LIVEKIT_URL
# Should output something like: wss://your-project.livekit.cloud
```

### File Format Issues
If you have non-WAV files, convert them first:
```bash
# Using ffmpeg
ffmpeg -i input.mp3 input.wav
```

### Dependencies
If you get import errors:
```bash
pip install --upgrade livekit livekit-plugins-noise-cancellation numpy
```

## Future Improvements

To make this a fully functional noise cancellation tool, the implementation would need:

1. **Stream Interception**: Capture the processed audio from LiveKit's internal pipeline
2. **Agent Framework**: Use LiveKit Agents for better pipeline integration  
3. **Real-time Processing**: Implement proper real-time audio processing
4. **Format Support**: Add support for more audio formats
5. **Batch Processing**: Support for processing multiple files

## Contributing

This is a demonstration tool. For production use, consider:
- Using LiveKit's official agent examples
- Implementing proper error handling
- Adding support for streaming audio processing
- Integration with LiveKit's full agent pipeline

## License

This tool is provided as-is for demonstration purposes. 