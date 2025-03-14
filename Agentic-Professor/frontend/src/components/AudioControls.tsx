import React, { useState, useRef, useEffect } from 'react';
import { Box, IconButton, Slider, Typography } from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  VolumeUp as VolumeIcon,
  VolumeOff as MuteIcon,
} from '@mui/icons-material';

interface AudioControlsProps {
  audioUrl: string;
}

export const AudioControls: React.FC<AudioControlsProps> = ({ audioUrl }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [volume, setVolume] = useState(1);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('ended', handleEnded);
    };
  }, []);

  const togglePlay = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };

  const toggleMute = () => {
    if (!audioRef.current) return;

    audioRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleVolumeChange = (_event: Event, newValue: number | number[]) => {
    const volumeValue = newValue as number;
    if (!audioRef.current) return;

    audioRef.current.volume = volumeValue;
    setVolume(volumeValue);
  };

  const handleTimeChange = (_event: Event, newValue: number | number[]) => {
    const timeValue = newValue as number;
    if (!audioRef.current) return;

    audioRef.current.currentTime = timeValue;
    setCurrentTime(timeValue);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ width: '100%', mt: 2 }}>
      <audio ref={audioRef} src={audioUrl} />
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
        <IconButton onClick={togglePlay} size="small">
          {isPlaying ? <PauseIcon /> : <PlayIcon />}
        </IconButton>
        
        <Typography variant="caption" sx={{ mx: 1, minWidth: 45 }}>
          {formatTime(currentTime)}
        </Typography>
        
        <Slider
          size="small"
          value={currentTime}
          max={duration}
          onChange={handleTimeChange}
          sx={{ mx: 2 }}
        />
        
        <Typography variant="caption" sx={{ mx: 1, minWidth: 45 }}>
          {formatTime(duration)}
        </Typography>
        
        <IconButton onClick={toggleMute} size="small">
          {isMuted ? <MuteIcon /> : <VolumeIcon />}
        </IconButton>
        
        <Slider
          size="small"
          value={isMuted ? 0 : volume}
          max={1}
          step={0.1}
          onChange={handleVolumeChange}
          sx={{ width: 100, ml: 1 }}
        />
      </Box>
    </Box>
  );
};
