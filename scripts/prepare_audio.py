#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np

from vqwave.audio import load_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert audio files to .u16 format for training')
    parser.add_argument('src', help='Source directory containing audio files')
    parser.add_argument('dest', help='Destination directory for .u16 files')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate (default: 22050)')
    
    args = parser.parse_args()
    
    src_dir = args.src
    dest_dir = args.dest
    sample_rate = args.sample_rate
    
    if not os.path.isdir(src_dir):
        print(f'Error: Source directory does not exist: {src_dir}')
        sys.exit(1)
    
    if not os.path.isdir(dest_dir):
        print(f'Error: Destination directory does not exist: {dest_dir}')
        sys.exit(1)
    
    audio_extensions = ('.m4a', '.mp3', '.wav', '.flac', '.ogg')
    
    for f in os.listdir(src_dir):
        if not f.lower().endswith(audio_extensions):
            continue
        
        src_file = os.path.join(src_dir, f)
        dest_file = os.path.join(dest_dir, f'{f}.u16')
        
        if os.path.exists(dest_file):
            print(f'Skipping {f} (already exists)')
            continue
        
        print(f'Processing {f}...')
        try:
            x = np.uint16(load_audio(src_file, sample_rate, to_float=False))
            with open(dest_file, 'wb') as fd:
                fd.write(x.tobytes())
            print(f'  Saved to {dest_file}')
        except Exception as e:
            print(f'  Error processing {f}: {e}')

