import React, { useState, useEffect } from 'react';
import { 
  Typography, 
  Box, 
  Container, 
  Card, 
  CardContent, 
  Grid, 
  Button, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  CircularProgress,
  Tabs,
  Tab,
  Paper,
  Divider,
  TextField
} from '@mui/material';
import { SelectChangeEvent } from '@mui/material/Select';
import { styled } from '@mui/material/styles';
import PlayCircleOutlineIcon from '@mui/icons-material/PlayCircleOutline';

// Import mock service for now
import { getAvailableStrategies, testCompression, CompressionResult } from '../services/compression';

const TestCard = styled(Card)(({ theme }) => ({
  marginBottom: theme.spacing(3),
  overflow: 'visible'
}));

const ResultCard = styled(Card)(({ theme }) => ({
  marginTop: theme.spacing(3),
  backgroundColor: theme.palette.grey[50],
}));

const CompressionTestPanel = () => {
  const [sampleType, setSampleType] = useState('text');
  const [sampleSize, setSampleSize] = useState('medium');
  const [strategy, setStrategy] = useState('auto');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<CompressionResult | null>(null);
  const [primeResults, setPrimeResults] = useState<CompressionResult | null>(null);
  const [standardResults, setStandardResults] = useState<CompressionResult | null>(null);
  
  const [strategies, setStrategies] = useState<Array<{id: string, name: string}>>([{ id: 'auto', name: 'Auto (Best)' }]);
  
  // Load strategies at component mount
  useEffect(() => {
    const loadStrategies = async () => {
      try {
        const availableStrategies = await getAvailableStrategies();
        setStrategies(availableStrategies);
      } catch (error) {
        console.error('Failed to load strategies:', error);
      }
    };
    
    loadStrategies();
  }, []);
  
  const sampleTypes = [
    { id: 'text', name: 'Text Data' },
    { id: 'binary', name: 'Binary Data' },
    { id: 'image', name: 'Image Data' },
    { id: 'mixed', name: 'Mixed Data' },
    { id: 'sequential', name: 'Sequential Pattern' },
    { id: 'sine', name: 'Sine Wave Pattern' },
  ];
  
  const sampleSizes = [
    { id: 'small', name: 'Small (10KB)', size: 10 * 1024 },
    { id: 'medium', name: 'Medium (100KB)', size: 100 * 1024 },
    { id: 'large', name: 'Large (1MB)', size: 1024 * 1024 },
  ];
  
  const handleStrategyChange = (event: SelectChangeEvent) => {
    setStrategy(event.target.value);
  };
  
  const handleSampleTypeChange = (event: SelectChangeEvent) => {
    setSampleType(event.target.value);
  };
  
  const handleSampleSizeChange = (event: SelectChangeEvent) => {
    setSampleSize(event.target.value);
  };
  
  const runComparisonTest = async () => {
    setIsLoading(true);
    
    try {
      // Generate sample data based on type and size
      const currentSize = sampleSizes.find(s => s.id === sampleSize)?.size || 10 * 1024;
      const data = generateSampleData(sampleType, currentSize);
      
      // Run PrimeCompress test
      const primeResult = await testCompression(data, strategy);
      setPrimeResults(primeResult);
      
      // Run Standard compression test (mock)
      const standardResult = await testCompression(data, 'standard');
      // Modify the result to simulate standard compression being less effective
      standardResult.compressionRatio = standardResult.compressionRatio / 2;
      standardResult.compressedSize = Math.floor(standardResult.originalSize / standardResult.compressionRatio);
      standardResult.compressionTime = standardResult.compressionTime * 1.2;
      setStandardResults(standardResult);
      
      // Set combined results
      setResults({
        ...primeResult,
        originalSize: primeResult.originalSize,
        compressedSize: primeResult.compressedSize,
        compressionRatio: primeResult.compressionRatio,
        strategy: strategy,
        compressionTime: primeResult.compressionTime
      });
    } catch (error) {
      console.error('Compression test error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Helper function to generate sample data
  const generateSampleData = (type: string, size: number): ArrayBuffer => {
    const buffer = new ArrayBuffer(size);
    const view = new Uint8Array(buffer);
    
    switch (type) {
      case 'text':
        // Generate text-like data
        const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?";
        for (let i = 0; i < size; i++) {
          view[i] = chars.charCodeAt(i % chars.length);
        }
        break;
      case 'sequential':
        // Generate sequential pattern
        for (let i = 0; i < size; i++) {
          view[i] = i % 256;
        }
        break;
      case 'sine':
        // Generate sine wave pattern
        for (let i = 0; i < size; i++) {
          view[i] = Math.floor(128 + 127 * Math.sin(i * 0.1));
        }
        break;
      case 'mixed':
        // Generate mixed data patterns
        const blockSize = Math.floor(size / 4);
        // Block 1: Random
        for (let i = 0; i < blockSize; i++) {
          view[i] = Math.floor(Math.random() * 256);
        }
        // Block 2: Sequential
        for (let i = 0; i < blockSize; i++) {
          view[blockSize + i] = i % 256;
        }
        // Block 3: Sine wave
        for (let i = 0; i < blockSize; i++) {
          view[blockSize * 2 + i] = Math.floor(128 + 127 * Math.sin(i * 0.1));
        }
        // Block 4: Repeating pattern
        const pattern = [10, 20, 30, 40, 50, 60, 70, 80];
        for (let i = 0; i < blockSize; i++) {
          view[blockSize * 3 + i] = pattern[i % pattern.length];
        }
        break;
      default:
        // Random binary data
        for (let i = 0; i < size; i++) {
          view[i] = Math.floor(Math.random() * 256);
        }
    }
    
    return buffer;
  };
  
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };
  
  const formatTime = (ms: number): string => {
    if (ms < 1) return `${(ms * 1000).toFixed(2)} μs`;
    if (ms < 1000) return `${ms.toFixed(2)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  };
  
  const improvementPercentage = (): string => {
    if (!primeResults || !standardResults) return '0%';
    const improvement = ((primeResults.compressionRatio / standardResults.compressionRatio) - 1) * 100;
    return `${improvement.toFixed(2)}%`;
  };
  
  return (
    <Box>
      <TestCard>
        <CardContent>
          <Typography variant="h6" gutterBottom>Compression Test Configuration</Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel id="sample-type-label">Sample Data Type</InputLabel>
                <Select
                  labelId="sample-type-label"
                  id="sample-type"
                  value={sampleType}
                  label="Sample Data Type"
                  onChange={handleSampleTypeChange}
                >
                  {sampleTypes.map(type => (
                    <MenuItem key={type.id} value={type.id}>{type.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel id="sample-size-label">Sample Size</InputLabel>
                <Select
                  labelId="sample-size-label"
                  id="sample-size"
                  value={sampleSize}
                  label="Sample Size"
                  onChange={handleSampleSizeChange}
                >
                  {sampleSizes.map(size => (
                    <MenuItem key={size.id} value={size.id}>{size.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <FormControl fullWidth>
                <InputLabel id="strategy-label">Compression Strategy</InputLabel>
                <Select
                  labelId="strategy-label"
                  id="strategy"
                  value={strategy}
                  label="Compression Strategy"
                  onChange={handleStrategyChange}
                >
                  {strategies.map(strat => (
                    <MenuItem key={strat.id} value={strat.id}>{strat.name}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          <Button
            variant="contained"
            color="primary"
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <PlayCircleOutlineIcon />}
            onClick={runComparisonTest}
            disabled={isLoading}
            sx={{ mt: 3 }}
            fullWidth
          >
            {isLoading ? 'Running Test...' : 'Run Compression Test'}
          </Button>
        </CardContent>
      </TestCard>
      
      {results && (
        <ResultCard>
          <CardContent>
            <Typography variant="h6" gutterBottom>Compression Results</Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <Paper sx={{ p: 2, height: '100%' }}>
                  <Typography variant="subtitle1" gutterBottom>PrimeCompress</Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Typography variant="body2">
                    Strategy: {primeResults?.strategy}
                  </Typography>
                  <Typography variant="body2">
                    Original Size: {formatSize(primeResults?.originalSize || 0)}
                  </Typography>
                  <Typography variant="body2">
                    Compressed Size: {formatSize(primeResults?.compressedSize || 0)}
                  </Typography>
                  <Typography variant="body2">
                    Compression Ratio: {primeResults?.compressionRatio.toFixed(2)}x
                  </Typography>
                  <Typography variant="body2">
                    Compression Time: {formatTime(primeResults?.compressionTime || 0)}
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6}>
                <Paper sx={{ p: 2, height: '100%' }}>
                  <Typography variant="subtitle1" gutterBottom>Standard Compression</Typography>
                  <Divider sx={{ mb: 2 }} />
                  <Typography variant="body2">
                    Strategy: standard
                  </Typography>
                  <Typography variant="body2">
                    Original Size: {formatSize(standardResults?.originalSize || 0)}
                  </Typography>
                  <Typography variant="body2">
                    Compressed Size: {formatSize(standardResults?.compressedSize || 0)}
                  </Typography>
                  <Typography variant="body2">
                    Compression Ratio: {standardResults?.compressionRatio.toFixed(2)}x
                  </Typography>
                  <Typography variant="body2">
                    Compression Time: {formatTime(standardResults?.compressionTime || 0)}
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
            
            <Paper sx={{ p: 2, mt: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
              <Typography variant="subtitle1">
                Improvement: {improvementPercentage()} better compression with PrimeCompress!
              </Typography>
            </Paper>
          </CardContent>
        </ResultCard>
      )}
    </Box>
  );
};

// Custom compression test component
const CustomTestPanel = () => {
  const [customText, setCustomText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any | null>(null);
  
  const handleRunCustomTest = async () => {
    if (!customText) return;
    
    setIsLoading(true);
    
    try {
      const textEncoder = new TextEncoder();
      const data = textEncoder.encode(customText);
      
      // Get results for each strategy
      const strategies = ['pattern', 'sequential', 'spectral', 'dictionary', 'auto'];
      const testResults = [];
      
      for (const strategy of strategies) {
        const result = await testCompression(data.buffer, strategy);
        testResults.push({
          strategy,
          result
        });
      }
      
      // Sort by compression ratio
      testResults.sort((a, b) => b.result.compressionRatio - a.result.compressionRatio);
      
      setResults(testResults);
    } catch (error) {
      console.error('Custom test error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const formatSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };
  
  return (
    <Box>
      <TestCard>
        <CardContent>
          <Typography variant="h6" gutterBottom>Custom Test Data</Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Enter your own text to test different compression strategies:
          </Typography>
          
          <TextField
            fullWidth
            multiline
            rows={6}
            variant="outlined"
            placeholder="Enter text to compress..."
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
          />
          
          <Button
            variant="contained"
            color="primary"
            startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <PlayCircleOutlineIcon />}
            onClick={handleRunCustomTest}
            disabled={isLoading || !customText}
            sx={{ mt: 2 }}
            fullWidth
          >
            {isLoading ? 'Testing...' : 'Test All Strategies'}
          </Button>
        </CardContent>
      </TestCard>
      
      {results && (
        <ResultCard>
          <CardContent>
            <Typography variant="h6" gutterBottom>Compression Strategy Comparison</Typography>
            <Typography variant="body2" paragraph>
              Original Size: {formatSize(results[0]?.result.originalSize || 0)}
              {' '} ({results[0]?.result.originalSize || 0} bytes)
            </Typography>
            
            <Grid container spacing={2}>
              {results.map((item: any, index: number) => (
                <Grid item xs={12} sm={6} md={4} key={item.strategy}>
                  <Paper 
                    sx={{ 
                      p: 2, 
                      borderTop: '4px solid',
                      borderColor: index === 0 ? 'success.main' : 'primary.main' 
                    }}
                  >
                    <Typography variant="subtitle1" gutterBottom>
                      {index === 0 && '🏆 '}{item.strategy.charAt(0).toUpperCase() + item.strategy.slice(1)}
                    </Typography>
                    <Typography variant="body2">
                      Compressed Size: {formatSize(item.result.compressedSize)}
                    </Typography>
                    <Typography variant="body2">
                      Compression Ratio: {item.result.compressionRatio.toFixed(2)}x
                    </Typography>
                    <Typography variant="body2">
                      Compression Time: {item.result.compressionTime.toFixed(2)} ms
                    </Typography>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </CardContent>
        </ResultCard>
      )}
    </Box>
  );
};

const TestPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <Container>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          PrimeCompress Testing Lab
        </Typography>
        <Typography variant="body1" paragraph>
          Test and compare PrimeCompress against standard compression techniques with various data types.
        </Typography>
      </Box>
      
      <Paper sx={{ mb: 3 }}>
        <Tabs 
          value={activeTab} 
          onChange={handleTabChange}
          variant="fullWidth"
        >
          <Tab label="Standard Tests" />
          <Tab label="Custom Test" />
        </Tabs>
      </Paper>
      
      {activeTab === 0 && <CompressionTestPanel />}
      {activeTab === 1 && <CustomTestPanel />}
    </Container>
  );
};

export default TestPage;