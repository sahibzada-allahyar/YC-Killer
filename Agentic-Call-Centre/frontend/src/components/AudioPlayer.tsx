'use client';

import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import {
  PlayIcon,
  PauseIcon,
  BackwardIcon,
  ForwardIcon,
  SpeakerWaveIcon,
  ArrowDownTrayIcon,
} from '@heroicons/react/24/solid';

interface AudioPlayerProps {
  url: string;
  onTimeUpdate?: (time: number) => void;
}

export default function AudioPlayer({ url, onTimeUpdate }: AudioPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const [playbackRate, setPlaybackRate] = useState(1);

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize WaveSurfer
    const wavesurfer = WaveSurfer.create({
      container: containerRef.current,
      waveColor: '#4B5563',
      progressColor: '#3B82F6',
      cursorColor: '#1D4ED8',
      barWidth: 2,
      barRadius: 3,
      height: 60,
      normalize: true,
      fillParent: true,
      minPxPerSec: 50,
    });

    // Load audio file
    wavesurfer.load(url);

    // Event listeners
    wavesurfer.on('ready', () => {
      wavesurferRef.current = wavesurfer;
      setDuration(wavesurfer.getDuration());
    });

    wavesurfer.on('audioprocess', (time) => {
      setCurrentTime(time);
      onTimeUpdate?.(time);
    });

    wavesurfer.on('finish', () => {
      setIsPlaying(false);
    });

    // Cleanup
    return () => {
      wavesurfer.destroy();
    };
  }, [url, onTimeUpdate]);

  const togglePlayPause = () => {
    if (!wavesurferRef.current) return;
    
    if (isPlaying) {
      wavesurferRef.current.pause();
    } else {
      wavesurferRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const seek = (direction: 'forward' | 'backward') => {
    if (!wavesurferRef.current) return;
    
    const currentTime = wavesurferRef.current.getCurrentTime();
    const newTime = direction === 'forward' 
      ? Math.min(currentTime + 10, duration)
      : Math.max(currentTime - 10, 0);
    
    wavesurferRef.current.seekTo(newTime / duration);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    if (wavesurferRef.current) {
      wavesurferRef.current.setVolume(newVolume);
    }
  };

  const handlePlaybackRateChange = (rate: number) => {
    setPlaybackRate(rate);
    if (wavesurferRef.current) {
      wavesurferRef.current.setPlaybackRate(rate);
    }
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleDownload = async () => {
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = downloadUrl;
      a.download = 'call-recording.mp3';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(downloadUrl);
      document.body.removeChild(a);
    } catch (error) {
      console.error('Error downloading audio:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow p-4">
      {/* Waveform */}
      <div ref={containerRef} className="mb-4" />

      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          {/* Play/Pause */}
          <button
            onClick={togglePlayPause}
            className="p-2 rounded-full hover:bg-gray-100"
          >
            {isPlaying ? (
              <PauseIcon className="h-6 w-6 text-gray-700" />
            ) : (
              <PlayIcon className="h-6 w-6 text-gray-700" />
            )}
          </button>

          {/* Seek Buttons */}
          <button
            onClick={() => seek('backward')}
            className="p-2 rounded-full hover:bg-gray-100"
          >
            <BackwardIcon className="h-5 w-5 text-gray-700" />
          </button>
          <button
            onClick={() => seek('forward')}
            className="p-2 rounded-full hover:bg-gray-100"
          >
            <ForwardIcon className="h-5 w-5 text-gray-700" />
          </button>

          {/* Time Display */}
          <div className="text-sm text-gray-600">
            {formatTime(currentTime)} / {formatTime(duration)}
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {/* Volume Control */}
          <div className="flex items-center space-x-2">
            <SpeakerWaveIcon className="h-5 w-5 text-gray-700" />
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={handleVolumeChange}
              className="w-20"
            />
          </div>

          {/* Playback Speed */}
          <select
            value={playbackRate}
            onChange={(e) => handlePlaybackRateChange(parseFloat(e.target.value))}
            className="text-sm border rounded px-2 py-1"
          >
            <option value="0.5">0.5x</option>
            <option value="1">1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
          </select>

          {/* Download Button */}
          <button
            onClick={handleDownload}
            className="p-2 rounded-full hover:bg-gray-100"
            title="Download recording"
          >
            <ArrowDownTrayIcon className="h-5 w-5 text-gray-700" />
          </button>
        </div>
      </div>
    </div>
  );
}
