import React, { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  LinearProgress,
  Paper,
  styled,
  CircularProgress,
  Alert,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import FileDownloadIcon from '@mui/icons-material/FileDownload';

// Compression service and hook
import useCompression from '../../hooks/useCompression';
import { getAvailableStrategies, Strategy } from '../../services/compression';
import { CompressionOptions } from '../../wasm/prime-compress-wasm';

interface FileUploadComponentProps {
  open: boolean;
  onClose: () => void;
}

const DropZone = styled(Paper)(({ theme }) => ({
  border: `2px dashed ${theme.palette.primary.main}`,
  borderRadius: theme.shape.borderRadius,
  padding: theme.spacing(3),
  textAlign: 'center',
  cursor: 'pointer',
  backgroundColor: theme.palette.background.default,
  '&:hover': {
    backgroundColor: theme.palette.action.hover,
  },
  marginBottom: theme.spacing(2),
}));

const ResultBox = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
  marginTop: theme.spacing(2),
  backgroundColor: theme.palette.grey[50],
}));

const FileUploadComponent: React.FC<FileUploadComponentProps> = ({ open, onClose }) => {
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [compressionComplete, setCompressionComplete] = useState(false);
  const [compressionResult, setCompressionResult] = useState<{
    originalSize: number;
    compressedSize: number;
    compressionRatio: number;
    strategy: string;
    compressionTime: number;
    compressedBlob?: Blob;
  } | null>(null);
  // Initialize with a default strategy (not an empty array)
  const [strategies, setStrategies] = useState<Strategy[]>([
    { id: 'auto', name: 'Auto (Best)' }
  ]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('auto');
  const [loadingStrategies, setLoadingStrategies] = useState(false);
  
  const { compressFile, isCompressing, error, resetState } = useCompression();

  // Load available compression strategies
  useEffect(() => {
    const loadStrategies = async () => {
      setLoadingStrategies(true);
      try {
        const availableStrategies = await getAvailableStrategies();
        // Make sure we have at least the default auto strategy
        if (availableStrategies && Array.isArray(availableStrategies) && availableStrategies.length > 0) {
          setStrategies(availableStrategies);
        } else {
          // Fallback to default strategies if none are returned
          setStrategies([
            { id: 'auto', name: 'Auto (Best)' }
          ]);
        }
      } catch (err) {
        console.error('Failed to load strategies:', err);
        // Set default strategies on error
        setStrategies([
          { id: 'auto', name: 'Auto (Best)' }
        ]);
      } finally {
        setLoadingStrategies(false);
      }
    };
    
    loadStrategies();
  }, []);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    setFile(file);
    resetState(); // Reset hook state
    // Reset component-specific states
    setCompressionComplete(false);
    setCompressionResult(null);
  };

  const handleStrategyChange = (event: SelectChangeEvent) => {
    setSelectedStrategy(event.target.value);
  };

  const handleCompress = async () => {
    if (!file) return;
    
    try {
      // Set up compression options
      const options: CompressionOptions = {};
      if (selectedStrategy !== 'auto') {
        options.strategy = selectedStrategy;
      }
      
      // Use the compression hook which uses WebAssembly internally
      const result = await compressFile(file, options);
      setCompressionResult(result);
      setCompressionComplete(true);
    } catch (err) {
      console.error('Error compressing file:', err);
      // Error is already handled by the hook
    }
  };

  const handleDownload = () => {
    if (!compressionResult?.compressedBlob) return;
    
    const url = URL.createObjectURL(compressionResult.compressedBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file?.name || 'file'}.compressed`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const resetComponentState = () => {
    setFile(null);
    setCompressionComplete(false);
    setCompressionResult(null);
  };

  const handleClose = () => {
    resetComponentState();
    resetState(); // Reset the hook state
    onClose();
  };

  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Compress File with PrimeCompress</DialogTitle>
      <DialogContent>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        <DropZone
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          sx={{
            backgroundColor: dragActive ? 'action.hover' : 'background.default',
          }}
        >
          <input
            type="file"
            id="file-upload"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-upload" style={{ cursor: 'pointer', width: '100%', height: '100%', display: 'block' }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 1 }} />
              <Typography variant="body1">
                {file ? file.name : 'Drag and drop a file here, or click to select a file'}
              </Typography>
              {file && (
                <Typography variant="body2" color="text.secondary">
                  Size: {formatSize(file.size)}
                </Typography>
              )}
            </Box>
          </label>
        </DropZone>
        
        {file && (
          <Box sx={{ mt: 2 }}>
            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel id="compression-strategy-label">Compression Strategy</InputLabel>
              <Select
                labelId="compression-strategy-label"
                id="compression-strategy"
                value={selectedStrategy}
                label="Compression Strategy"
                onChange={handleStrategyChange}
                disabled={isCompressing || loadingStrategies}
              >
                {strategies.map((strategy) => (
                  <MenuItem key={strategy.id} value={strategy.id}>
                    {strategy.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            
            <Button
              variant="contained"
              color="primary"
              fullWidth
              onClick={handleCompress}
              disabled={isCompressing}
              startIcon={isCompressing ? <CircularProgress size={20} color="inherit" /> : null}
            >
              {isCompressing ? 'Compressing...' : 'Compress File'}
            </Button>
          </Box>
        )}
        
        {isCompressing && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
          </Box>
        )}
        
        {compressionComplete && compressionResult && (
          <ResultBox>
            <Typography variant="h6" gutterBottom>Compression Results</Typography>
            <Typography variant="body1">
              Original size: {formatSize(compressionResult.originalSize)}
            </Typography>
            <Typography variant="body1">
              Compressed size: {formatSize(compressionResult.compressedSize)}
            </Typography>
            <Typography variant="body1">
              Compression ratio: {compressionResult.compressionRatio.toFixed(2)}x
            </Typography>
            <Typography variant="body1" gutterBottom>
              Strategy used: {compressionResult.strategy}
            </Typography>
            <Typography variant="body1" gutterBottom>
              Time taken: {compressionResult.compressionTime.toFixed(2)} ms
            </Typography>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<FileDownloadIcon />}
              onClick={handleDownload}
              fullWidth
              sx={{ mt: 1 }}
            >
              Download Compressed File
            </Button>
          </ResultBox>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
};

export default FileUploadComponent;